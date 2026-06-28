from .client import PageIndexClient
from .page_index_md import md_to_tree, markdown_to_tree
from .retrieve import get_document, get_document_structure, get_page_content

__all__ = [
    "PageIndexClient",
    "get_document",
    "get_document_structure",
    "get_page_content",
    "markdown_to_tree",
    "md_to_tree",
    "page_index",
    "page_index_main",
]


def page_index(*args, **kwargs):
    from .page_index import page_index as page_index_fn

    return page_index_fn(*args, **kwargs)


def page_index_main(*args, **kwargs):
    from .page_index import page_index_main as page_index_main_fn

    return page_index_main_fn(*args, **kwargs)
