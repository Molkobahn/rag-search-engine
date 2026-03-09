#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_command,
    semantic_chunk_command,
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

    chunk_parser = subparsers.add_parser("chunk", help="Turn text into chunks")
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="Tunable word limit for chunks")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Tunable overlap arg")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Splits text into sentence chunks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Tunable max length for chunks")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Tunable overlap option for chunks")
    
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
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case _:
            parser.print_help()
        

if __name__ == "__main__":
    main()