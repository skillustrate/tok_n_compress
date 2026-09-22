"""
Command-Line Interface (CLI) for tok_n_compress.

Enables terminal-based agent harnesses, shell scripts, and CI/CD pipelines
to interact with the conversation memory compression engine directly.
"""

import sys
import json
import argparse
from typing import List, Optional

from .skill import ConversationCompressorSkill


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tok-compress",
        description="tok_n_compress: Hybrid Memory Model conversation compression and retrieval CLI"
    )
    parser.add_argument("--db", default="./checkpoints.db", help="Path to SQLite checkpoints database")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # Subcommand: compress
    compress_p = subparsers.add_parser("compress", help="Compress a conversation history JSON file or string")
    compress_p.add_argument("--file", "-f", help="Path to JSON file containing list of conversation messages")
    compress_p.add_argument("--input", "-i", help="Raw JSON string representing list of conversation messages")
    compress_p.add_argument("--keep-recent", "-k", type=int, default=2, help="Number of recent messages to preserve uncompressed")
    compress_p.add_argument("--query", "-q", help="Optional user query context")

    # Subcommand: query
    query_p = subparsers.add_parser("query", help="Query deep memory for relevant conversation segments")
    query_p.add_argument("query_text", help="Text, topic, or error code to search for")
    query_p.add_argument("--limit", "-l", type=int, default=3, help="Maximum number of results to return")

    # Subcommand: rehydrate
    rehydrate_p = subparsers.add_parser("rehydrate", help="Rehydrate a raw conversation segment into active context format")
    rehydrate_p.add_argument("target_id", type=int, help="Segment ID or Checkpoint ID to rehydrate")
    rehydrate_p.add_argument("--query", "-q", help="Optional user query context")

    # Subcommand: stats
    subparsers.add_parser("stats", help="Display memory and database statistics")

    # Subcommand: hierarchy
    hier_p = subparsers.add_parser("hierarchy", help="Display ancestor checkpoint lineage")
    hier_p.add_argument("checkpoint_id", type=int, help="Checkpoint ID to trace back to root")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    parser = create_parser()
    parsed = parser.parse_args(args)

    skill = ConversationCompressorSkill(db_path=parsed.db)

    if parsed.command == "compress":
        messages = []
        if parsed.file:
            with open(parsed.file, "r", encoding="utf-8") as f:
                messages = json.load(f)
        elif parsed.input:
            messages = json.loads(parsed.input)
        else:
            print("Error: Specify either --file or --input for compression.", file=sys.stderr)
            return 1

        if not isinstance(messages, list):
            print("Error: Input must be a JSON array of message objects.", file=sys.stderr)
            return 1

        result = skill.compress(
            raw_history=messages,
            user_query=parsed.query,
            keep_recent_n=parsed.keep_recent
        )
        print(json.dumps(result, indent=2))
        return 0

    elif parsed.command == "query":
        matches = skill.query(parsed.query_text, limit=parsed.limit)
        print(json.dumps(matches, indent=2))
        return 0

    elif parsed.command == "rehydrate":
        try:
            formatted = skill.rehydrate(parsed.target_id, user_query=parsed.query)
            print(formatted)
            return 0
        except Exception as e:
            print(f"Error rehydrating: {e}", file=sys.stderr)
            return 1

    elif parsed.command == "stats":
        stats = skill.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    elif parsed.command == "hierarchy":
        lineage = skill.get_hierarchy(parsed.checkpoint_id)
        print(json.dumps(lineage, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
