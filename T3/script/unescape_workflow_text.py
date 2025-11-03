"""Utility script to restore Chinese characters in workflow definition files.

This tool scans the given files (or directories) for unicode escape sequences
such as ``\u7edf`` and converts them back to their readable form. By default it
processes the ``T3/workflow`` directory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Iterator

# Supported workflow definition file suffixes.
WORKFLOW_SUFFIXES = {".json", ".yaml", ".yml", ".toon"}

# Matches one or more consecutive unicode escape tokens (e.g. "\\u4e2d\\u56fd").
UNICODE_ESCAPE_PATTERN = re.compile(r"(\\u[0-9a-fA-F]{4})+")


def iter_target_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield the files that should be processed."""
    for path in paths:
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in WORKFLOW_SUFFIXES:
                    yield candidate
        elif path.is_file():
            if path.suffix.lower() in WORKFLOW_SUFFIXES:
                yield path
        else:
            continue


def decode_unicode_escapes(text: str) -> str:
    """Replace unicode escape sequences with their literal characters."""

    def _replace(match: re.Match[str]) -> str:
        escape_sequence = match.group(0)
        try:
            return escape_sequence.encode("ascii").decode("unicode_escape")
        except UnicodeDecodeError:
            return escape_sequence

    return UNICODE_ESCAPE_PATTERN.sub(_replace, text)


def process_file(path: Path, dry_run: bool) -> bool:
    """Process a single file and return True when any change is made."""
    original = path.read_text(encoding="utf-8")
    updated = decode_unicode_escapes(original)

    if updated == original:
        return False

    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert unicode escapes back to Chinese characters in workflow files.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(__file__).resolve().parents[1] / "workflow"],
        help="Files or directories to process (defaults to T3/workflow).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report potential changes without modifying them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = list(iter_target_files(args.paths))

    if not targets:
        print("No matching workflow files found.")
        return

    changed_files = 0
    for file_path in targets:
        if process_file(file_path, args.dry_run):
            changed_files += 1
            print(f"Updated {file_path}")
        elif args.dry_run:
            print(f"No change needed for {file_path}")

    if args.dry_run:
        print(f"Dry run completed. {changed_files} file(s) would be updated.")
    else:
        print(f"Processing completed. {changed_files} file(s) updated.")


if __name__ == "__main__":
    main()
