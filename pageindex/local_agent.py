import asyncio
import json
import os
import re
import uuid

from .epub import epub_to_tree
from .page_index_md import md_to_tree
from .proxy_llm import last_error, llm_completion
from .pir import get_subtree_text, read_pir, render_compact_tree, search_nodes
from .retrieve import get_document, get_document_structure, get_page_content


SELECT_PROMPT = """You are a document retrieval agent.
You will receive document metadata, a tree structure, and a user question.
Choose the smallest set of relevant line numbers or page numbers to inspect.

Return only JSON:
{{
  "thinking": "short reason",
  "pages": "line/page range like 12,18-22"
}}

Document metadata:
{metadata}

Document tree:
{structure}

Question:
{question}
"""


ANSWER_PROMPT = """Answer the question using only the provided document excerpts.
If the excerpts are insufficient, say what is missing.
Mention relevant line/page references when useful.
Be concise and focus on facts that directly answer the question.

Question:
{question}

Document excerpts:
{content}
"""


PIR_SELECT_PROMPT = """You are a document retrieval agent.
You will receive document metadata, a compact PageIndex tree, and a user question.
Choose the smallest set of relevant node IDs to inspect.

Return only JSON:
{{
  "thinking": "short reason",
  "node_ids": ["0001", "0002"]
}}

Document metadata:
{metadata}

Compact tree:
{tree}

Question:
{question}
"""


def _extract_json(text):
    if not text:
        return {}
    match = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _first_available_reference(structure):
    if isinstance(structure, str):
        structure = json.loads(structure)
    stack = list(structure) if isinstance(structure, list) else [structure]
    while stack:
        node = stack.pop(0)
        if node.get("line_num"):
            return str(node["line_num"])
        if node.get("start_index"):
            end = node.get("end_index") or node["start_index"]
            return f"{node['start_index']}-{end}" if end != node["start_index"] else str(node["start_index"])
        stack.extend(node.get("nodes") or [])
    return "1"


async def _index_document(file_path, model):
    file_path = os.path.abspath(os.path.expanduser(file_path))
    ext = os.path.splitext(file_path)[1].lower()
    doc_id = str(uuid.uuid4())

    if ext in {".md", ".markdown"}:
        result = await md_to_tree(
            file_path,
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
            model=model,
        )
        doc_type = "md"
        line_count = result.get("line_count", 0)
    elif ext == ".epub":
        result = await epub_to_tree(
            file_path,
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
            model=model,
        )
        doc_type = "epub"
        line_count = result.get("line_count", 0)
    elif ext == ".pdf":
        from .page_index import page_index
        import PyPDF2

        result = page_index(
            doc=file_path,
            model=model,
            if_add_node_summary="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
            if_add_doc_description="no",
        )
        pages = []
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(pdf_reader.pages, 1):
                pages.append({"page": i, "content": page.extract_text() or ""})
        return doc_id, {
            doc_id: {
                "id": doc_id,
                "type": "pdf",
                "path": file_path,
                "doc_name": result.get("doc_name", ""),
                "doc_description": result.get("doc_description", ""),
                "page_count": len(pages),
                "structure": result["structure"],
                "pages": pages,
            }
        }
    else:
        raise ValueError(f"Unsupported file format for local agent: {file_path}")

    return doc_id, {
        doc_id: {
            "id": doc_id,
            "type": doc_type,
            "path": file_path,
            "doc_name": result.get("doc_name", ""),
            "doc_description": result.get("doc_description", ""),
            "line_count": line_count,
            "structure": result["structure"],
        }
    }


def ask_document(file_path, question, model=None, verbose=False):
    doc_id, documents = asyncio.run(_index_document(file_path, model))
    metadata = get_document(documents, doc_id)
    structure = get_document_structure(documents, doc_id)

    selection_raw = llm_completion(
        model,
        SELECT_PROMPT.format(metadata=metadata, structure=structure, question=question),
    )
    if not selection_raw:
        raise RuntimeError(
            "Local LLM did not return a response. Start llama.cpp server at "
            "http://127.0.0.1:8080/v1 or pass --model ollama/<model> / --model litellm/<model>."
        )
    selection = _extract_json(selection_raw)
    pages = selection.get("pages") or _first_available_reference(structure)
    content = get_page_content(documents, doc_id, pages)

    if verbose:
        print(f"[agent] selected pages/lines: {pages}")

    answer = llm_completion(
        model,
        ANSWER_PROMPT.format(question=question, content=content),
    )
    if not answer:
        detail = f" Last error: {last_error()}" if last_error() else ""
        raise RuntimeError(
            "Local LLM did not return an answer. Check that the selected model server is running."
            + detail
        )
    return answer


def _without_ancestor_duplicates(compiled, node_ids):
    node_by_id = {node["node_id"]: node for node in compiled["nodes"]}
    selected = [node_id for node_id in node_ids if node_id in node_by_id]
    selected_set = set(selected)
    redundant = set()

    for node_id in selected:
        parent = node_by_id[node_id].get("parent")
        while parent is not None:
            parent_id = compiled["nodes"][parent]["node_id"]
            if parent_id in selected_set:
                redundant.add(parent_id)
            parent = compiled["nodes"][parent].get("parent")

    return [node_id for node_id in selected if node_id not in redundant]


def _build_pir_excerpts(compiled, node_ids, per_node_chars=6000, total_chars=18000):
    excerpts = []
    total = 0
    for node_id in node_ids:
        remaining = total_chars - total
        if remaining <= 0:
            break
        text = get_subtree_text(compiled, node_id, max_chars=min(per_node_chars, remaining))
        if not text:
            continue
        excerpts.append(text)
        total += len(text)
    return excerpts, total


def ask_pir(pir_path, question, model=None, verbose=False):
    compiled = read_pir(pir_path)
    metadata = json.dumps(compiled.get("document", {}), ensure_ascii=False)
    tree = render_compact_tree(compiled, max_depth=2)
    lexical_node_ids = search_nodes(compiled, question, limit=4)
    node_ids = lexical_node_ids

    if not node_ids:
        selection_raw = llm_completion(
            model,
            PIR_SELECT_PROMPT.format(metadata=metadata, tree=tree, question=question),
        )
        if not selection_raw:
            raise RuntimeError(
                "Local LLM did not return a response. Start llama.cpp server at "
                "http://127.0.0.1:8080/v1 or pass --model ollama/<model> / --model litellm/<model>."
            )
        selection = _extract_json(selection_raw)
        node_ids = selection.get("node_ids") or selection.get("nodes") or []
        if isinstance(node_ids, str):
            node_ids = [part.strip() for part in node_ids.split(",") if part.strip()]

    node_ids = _without_ancestor_duplicates(compiled, list(dict.fromkeys(node_ids)))
    if not node_ids and compiled.get("nodes"):
        node_ids = [compiled["nodes"][0]["node_id"]]

    excerpts, excerpt_chars = _build_pir_excerpts(compiled, node_ids)
    if not excerpts:
        raise RuntimeError("PIR retrieval did not find text for the selected nodes.")

    if verbose:
        print(f"[agent] selected node_ids: {', '.join(node_ids)}")
        if lexical_node_ids:
            print(f"[agent] lexical node_ids: {', '.join(lexical_node_ids)}")
        print(f"[agent] excerpt chars: {excerpt_chars}")

    answer = llm_completion(
        model,
        ANSWER_PROMPT.format(question=question, content="\n\n".join(excerpts)),
    )
    if not answer:
        detail = f" Last error: {last_error()}" if last_error() else ""
        raise RuntimeError(
            "Local LLM did not return an answer. Check that the selected model server is running."
            + detail
        )
    return answer
