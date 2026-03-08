#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
)
from lib.search_utils import(
    DEFAULT_SEARCH_LIMIT
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser("verify", help="Verify version of Sentence Transformer model")

    embed_parser = subparsers.add_parser("embed_text", help="Embed input text")
    embed_parser.add_argument("text", type=str, help="Input string to embed")
    
    subparsers.add_parser("verify_embeddings", help="Verify embeddings of documents")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query to vector")
    embed_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Search for movie title and score")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Tunable result limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()
        

if __name__ == "__main__":
    main()