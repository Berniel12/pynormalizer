#!/usr/bin/env python3
"""
Script to run the tender normalizer.
"""
import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent))

from src.main import process_all_tenders, process_source


def main():
    parser = argparse.ArgumentParser(description="Run the tender normalizer")
    parser.add_argument(
        "--source", "-s", type=str, help="Source table name (omit to process all sources)"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=25, help="Maximum number of tenders to process per source"
    )
    args = parser.parse_args()

    if args.source:
        # Process only the specified source
        import asyncio
        asyncio.run(process_source(args.source, args.limit))
    else:
        # Process all sources
        process_all_tenders(args.limit)


if __name__ == "__main__":
    main() 