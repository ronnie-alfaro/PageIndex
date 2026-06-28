import asyncio
import json
import os
import re
import uuid

from .epub import epub_to_tree
from .page_index_md import md_to_tree
from .proxy_llm import llm_completion
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

Question:
{question}

Document excerpts:
{content}
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
        raise RuntimeError(
            "Local LLM did not return an answer. Check that the selected model server is running."
        )
    return answer
