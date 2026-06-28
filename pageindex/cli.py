import argparse
import asyncio
import json
import os
import sys

from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader


def _build_ask_parser():
    parser = argparse.ArgumentParser(description="Ask a local agent questions about a document")
    parser.add_argument("question", help="Question to answer from the document")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--md_path", type=str, help="Path to the Markdown file")
    parser.add_argument("--epub_path", type=str, help="Path to the EPUB file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use. Defaults to local/default via llama.cpp at 127.0.0.1:8080/v1",
    )
    parser.add_argument("--verbose", action="store_true", help="Print agent retrieval decisions")
    return parser


def _build_parser():
    parser = argparse.ArgumentParser(description="Process PDF or Markdown document and generate structure")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--md_path", type=str, help="Path to the Markdown file")
    parser.add_argument("--epub_path", type=str, help="Path to the EPUB file")

    parser.add_argument("--model", type=str, default=None, help="Model to use (overrides config.yaml)")

    parser.add_argument(
        "--toc-check-pages",
        type=int,
        default=None,
        help="Number of pages to check for table of contents (PDF only)",
    )
    parser.add_argument(
        "--max-pages-per-node",
        type=int,
        default=None,
        help="Maximum number of pages per node (PDF only)",
    )
    parser.add_argument(
        "--max-tokens-per-node",
        type=int,
        default=None,
        help="Maximum number of tokens per node (PDF only)",
    )

    parser.add_argument("--if-add-node-id", type=str, default=None, help="Whether to add node id to the node")
    parser.add_argument("--if-add-node-summary", type=str, default=None, help="Whether to add summary to the node")
    parser.add_argument("--if-add-doc-description", type=str, default=None, help="Whether to add doc description to the doc")
    parser.add_argument("--if-add-node-text", type=str, default=None, help="Whether to add text to the node")

    parser.add_argument(
        "--if-thinning",
        type=str,
        default="no",
        help="Whether to apply tree thinning for markdown (markdown only)",
    )
    parser.add_argument(
        "--thinning-threshold",
        type=int,
        default=5000,
        help="Minimum token threshold for thinning (markdown only)",
    )
    parser.add_argument(
        "--summary-token-threshold",
        type=int,
        default=200,
        help="Token threshold for generating summaries (markdown only)",
    )
    return parser


def _selected_path(args):
    paths = [path for path in (args.pdf_path, args.md_path, args.epub_path) if path]
    if not paths:
        raise ValueError("One of --pdf_path, --md_path, or --epub_path must be specified")
    if len(paths) > 1:
        raise ValueError("Only one of --pdf_path, --md_path, or --epub_path can be specified")
    return paths[0]


def _validate_args(args):
    _selected_path(args)


def _write_result(result, source_path):
    name = os.path.splitext(os.path.basename(source_path))[0]
    output_dir = "./results"
    output_file = f"{output_dir}/{name}_structure.json"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Tree structure saved to: {output_file}")


def _process_pdf(args):
    if not args.pdf_path.lower().endswith(".pdf"):
        raise ValueError("PDF file must have .pdf extension")
    if not os.path.isfile(args.pdf_path):
        raise ValueError(f"PDF file not found: {args.pdf_path}")

    from pageindex.page_index import page_index_main

    user_opt = {
        "model": args.model,
        "toc_check_page_num": args.toc_check_pages,
        "max_page_num_each_node": args.max_pages_per_node,
        "max_token_num_each_node": args.max_tokens_per_node,
        "if_add_node_id": args.if_add_node_id,
        "if_add_node_summary": args.if_add_node_summary,
        "if_add_doc_description": args.if_add_doc_description,
        "if_add_node_text": args.if_add_node_text,
    }
    opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

    result = page_index_main(args.pdf_path, opt)
    print("Parsing done, saving to file...")
    _write_result(result, args.pdf_path)


def _process_markdown(args):
    if not args.md_path.lower().endswith((".md", ".markdown")):
        raise ValueError("Markdown file must have .md or .markdown extension")
    if not os.path.isfile(args.md_path):
        raise ValueError(f"Markdown file not found: {args.md_path}")

    print("Processing markdown file...")

    user_opt = {
        "model": args.model,
        "if_add_node_summary": args.if_add_node_summary,
        "if_add_doc_description": args.if_add_doc_description,
        "if_add_node_text": args.if_add_node_text,
        "if_add_node_id": args.if_add_node_id,
    }
    opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

    result = asyncio.run(
        md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == "yes",
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id,
        )
    )

    print("Parsing done, saving to file...")
    _write_result(result, args.md_path)


def _process_epub(args):
    if not args.epub_path.lower().endswith(".epub"):
        raise ValueError("EPUB file must have .epub extension")
    if not os.path.isfile(args.epub_path):
        raise ValueError(f"EPUB file not found: {args.epub_path}")

    from pageindex.epub import epub_to_tree

    print("Processing EPUB file...")

    user_opt = {
        "model": args.model,
        "if_add_node_summary": args.if_add_node_summary,
        "if_add_doc_description": args.if_add_doc_description,
        "if_add_node_text": args.if_add_node_text,
        "if_add_node_id": args.if_add_node_id,
    }
    opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

    result = asyncio.run(
        epub_to_tree(
            epub_path=args.epub_path,
            if_thinning=args.if_thinning.lower() == "yes",
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id,
        )
    )

    print("Parsing done, saving to file...")
    _write_result(result, args.epub_path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ask":
        from pageindex.local_agent import ask_document

        parser = _build_ask_parser()
        args = parser.parse_args(argv[1:])
        file_path = _selected_path(args)
        try:
            answer = ask_document(file_path, args.question, model=args.model, verbose=args.verbose)
        except RuntimeError as e:
            parser.exit(1, f"error: {e}\n")
        print(answer)
        return

    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    if args.pdf_path:
        _process_pdf(args)
    elif args.md_path:
        _process_markdown(args)
    else:
        _process_epub(args)


if __name__ == "__main__":
    main()
