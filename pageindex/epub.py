import os
import posixpath
import zipfile
from html import unescape
from html.parser import HTMLParser
from xml.etree import ElementTree

from .page_index_md import markdown_to_tree


CONTAINER_PATH = "META-INF/container.xml"
OCF_NS = {"ocf": "urn:oasis:names:tc:opendocument:xmlns:container"}
OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}


class _HtmlToMarkdownParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.heading_level = None
        self.skip_depth = 0
        self.in_list_item = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._paragraph_break()
            self.heading_level = int(tag[1])
            self.parts.append("#" * self.heading_level + " ")
        elif tag in {"p", "section", "article", "div"}:
            self._paragraph_break()
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self._paragraph_break()
            self.in_list_item = True
            self.parts.append("- ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.heading_level = None
            self._paragraph_break()
        elif tag in {"p", "section", "article", "div", "li", "ul", "ol"}:
            self.in_list_item = False
            self._paragraph_break()

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = " ".join(unescape(data).split())
        if not text:
            return
        if self.parts and not self.parts[-1].endswith((" ", "\n", "# ", "## ", "### ", "#### ", "##### ", "###### ", "- ")):
            self.parts.append(" ")
        self.parts.append(text)

    def markdown(self):
        text = "".join(self.parts)
        lines = [line.rstrip() for line in text.splitlines()]
        compact = []
        blank = False
        for line in lines:
            if not line.strip():
                if not blank:
                    compact.append("")
                blank = True
            else:
                compact.append(line)
                blank = False
        return "\n".join(compact).strip()

    def _paragraph_break(self):
        if self.parts and not "".join(self.parts[-2:]).endswith("\n\n"):
            self.parts.append("\n\n")


def _read_rootfile_path(epub_zip):
    container = ElementTree.fromstring(epub_zip.read(CONTAINER_PATH))
    rootfile = container.find(".//ocf:rootfile", OCF_NS)
    if rootfile is None:
        raise ValueError("EPUB container does not define a rootfile")
    return rootfile.attrib["full-path"]


def _read_opf(epub_zip, opf_path):
    opf = ElementTree.fromstring(epub_zip.read(opf_path))
    base_dir = posixpath.dirname(opf_path)

    manifest = {}
    for item in opf.findall(".//opf:manifest/opf:item", OPF_NS):
        item_id = item.attrib.get("id")
        href = item.attrib.get("href")
        if item_id and href:
            manifest[item_id] = {
                "href": posixpath.normpath(posixpath.join(base_dir, href)),
                "media_type": item.attrib.get("media-type", ""),
            }

    spine = []
    for itemref in opf.findall(".//opf:spine/opf:itemref", OPF_NS):
        idref = itemref.attrib.get("idref")
        if idref in manifest:
            spine.append(manifest[idref])

    title_node = opf.find(".//dc:title", OPF_NS)
    title = title_node.text.strip() if title_node is not None and title_node.text else None
    return title, spine


def _html_to_markdown(html):
    parser = _HtmlToMarkdownParser()
    parser.feed(html)
    parser.close()
    return parser.markdown()


def epub_to_markdown(epub_path):
    with zipfile.ZipFile(epub_path) as epub_zip:
        opf_path = _read_rootfile_path(epub_zip)
        title, spine = _read_opf(epub_zip, opf_path)
        sections = []

        for item in spine:
            media_type = item["media_type"]
            if media_type not in {"application/xhtml+xml", "text/html"}:
                continue
            raw = epub_zip.read(item["href"])
            html = raw.decode("utf-8", errors="replace")
            markdown = _html_to_markdown(html)
            if markdown:
                sections.append(markdown)

    doc_name = title or os.path.splitext(os.path.basename(epub_path))[0]
    markdown = "\n\n".join(sections).strip()
    if markdown and not markdown.lstrip().startswith("#"):
        markdown = f"# {doc_name}\n\n{markdown}"
    return doc_name, markdown


async def epub_to_tree(
    epub_path,
    if_thinning=False,
    min_token_threshold=None,
    if_add_node_summary='no',
    summary_token_threshold=None,
    model=None,
    if_add_doc_description='no',
    if_add_node_text='no',
    if_add_node_id='yes',
):
    doc_name, markdown_content = epub_to_markdown(epub_path)
    if not markdown_content:
        raise ValueError(f"No readable HTML content found in EPUB: {epub_path}")
    result = await markdown_to_tree(
        markdown_content=markdown_content,
        doc_name=doc_name,
        if_thinning=if_thinning,
        min_token_threshold=min_token_threshold,
        if_add_node_summary=if_add_node_summary,
        summary_token_threshold=summary_token_threshold,
        model=model,
        if_add_doc_description=if_add_doc_description,
        if_add_node_text=if_add_node_text,
        if_add_node_id=if_add_node_id,
    )
    result["type"] = "epub"
    return result
