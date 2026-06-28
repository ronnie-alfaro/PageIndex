import os
import re
import unicodedata
import zlib

import msgpack

PIR_MAGIC = "PIR1"
PIR_HEADER = b"PIR1Z"
STOPWORDS = {
    "about", "como", "con", "cual", "cuando", "de", "del", "dice", "el", "en",
    "es", "esta", "este", "la", "las", "lo", "los", "por", "que", "sobre",
    "the", "what", "where", "who",
}


def _normalize(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower()


def _query_terms(query):
    terms = re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", _normalize(query))
    return [term for term in terms if term not in STOPWORDS]


def _walk_nodes(nodes, parent_index=None, depth=0, flat=None):
    if flat is None:
        flat = []

    previous_sibling = None
    first_child_by_parent = {}
    for order, node in enumerate(nodes):
        index = len(flat)
        text = node.get("text", "")
        record = {
            "index": index,
            "node_id": node.get("node_id") or str(index).zfill(4),
            "parent": parent_index,
            "first_child": None,
            "next_sibling": None,
            "depth": depth,
            "order": order,
            "title": node.get("title", ""),
            "summary": node.get("summary") or node.get("prefix_summary") or "",
            "line_num": node.get("line_num"),
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
            "text_index": index if text else None,
            "text_len": len(text),
        }
        flat.append(record)

        if parent_index is not None and parent_index not in first_child_by_parent:
            first_child_by_parent[parent_index] = index
            flat[parent_index]["first_child"] = index
        if previous_sibling is not None:
            flat[previous_sibling]["next_sibling"] = index
        previous_sibling = index

        children = node.get("nodes") or []
        if children:
            _walk_nodes(children, parent_index=index, depth=depth + 1, flat=flat)

    return flat


def compile_tree(result):
    structure = result.get("structure") or []
    nodes = _walk_nodes(structure)
    texts = []
    node_by_id = {}
    children_by_index = {}
    line_to_index = {}

    for node in nodes:
        node_by_id[node["node_id"]] = node["index"]
        parent = node["parent"]
        if parent is not None:
            children_by_index.setdefault(parent, []).append(node["index"])
        if node.get("line_num") is not None:
            line_to_index[str(node["line_num"])] = node["index"]
        texts.append("")

    def fill_texts(tree_nodes):
        for node in tree_nodes:
            node_id = node.get("node_id")
            if node_id in node_by_id:
                texts[node_by_id[node_id]] = node.get("text", "")
            fill_texts(node.get("nodes") or [])

    fill_texts(structure)

    return {
        "magic": PIR_MAGIC,
        "version": 1,
        "document": {
            "doc_name": result.get("doc_name", ""),
            "doc_description": result.get("doc_description", ""),
            "type": result.get("type", ""),
            "line_count": result.get("line_count"),
            "page_count": result.get("page_count"),
        },
        "nodes": nodes,
        "texts": texts,
        "indexes": {
            "node_by_id": node_by_id,
            "children_by_index": children_by_index,
            "line_to_index": line_to_index,
        },
    }


def write_pir(result, output_path):
    compiled = compile_tree(result)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        payload = msgpack.packb(compiled, use_bin_type=True)
        f.write(PIR_HEADER)
        f.write(zlib.compress(payload, level=6))
    return output_path


def read_pir(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(PIR_HEADER):
        raw = zlib.decompress(raw[len(PIR_HEADER):])
    data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    if data.get("magic") != PIR_MAGIC:
        raise ValueError(f"Invalid PIR file: {path}")
    return data


def get_node(compiled, node_id):
    index = compiled["indexes"]["node_by_id"].get(node_id)
    if index is None:
        return None
    return compiled["nodes"][index]


def get_children(compiled, node_id):
    node = get_node(compiled, node_id)
    if not node:
        return []
    children = compiled["indexes"]["children_by_index"].get(node["index"], [])
    return [compiled["nodes"][child] for child in children]


def get_text(compiled, node_id):
    node = get_node(compiled, node_id)
    if not node or node.get("text_index") is None:
        return ""
    return compiled["texts"][node["text_index"]]


def get_subtree_text(compiled, node_id, max_chars=24000):
    root = get_node(compiled, node_id)
    if not root:
        return ""

    root_depth = root["depth"]
    start = root["index"]
    parts = []
    total = 0

    for node in compiled["nodes"][start:]:
        if node["index"] != start and node["depth"] <= root_depth:
            break
        text = get_text(compiled, node["node_id"])
        if not text:
            continue
        chunk = f"[node {node['node_id']}] {node['title']}\n{text}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


def search_nodes(compiled, query, limit=5):
    terms = _query_terms(query)
    if not terms:
        return []

    scored = []
    for node in compiled["nodes"]:
        title = _normalize(node.get("title", ""))
        text = _normalize(get_text(compiled, node["node_id"]))
        score = 0
        for term in terms:
            if term in title:
                score += 10
            if term in text:
                score += min(5, text.count(term))
        if score:
            scored.append((score, node["depth"], node["index"], node["node_id"]))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [node_id for _, _, _, node_id in scored[:limit]]


def render_compact_tree(compiled, max_depth=2):
    lines = []
    for node in compiled["nodes"]:
        if node["depth"] > max_depth:
            continue
        ref = node.get("line_num") or node.get("start_index") or ""
        suffix = f" ({ref})" if ref else ""
        lines.append(f"{'  ' * node['depth']}[{node['node_id']}] {node['title']}{suffix}")
    return "\n".join(lines)
