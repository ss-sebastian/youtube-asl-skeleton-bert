"""Aggregate real/shuffled lexical tokens within exact ASL-LEX codes."""

from __future__ import annotations

import argparse
from pathlib import Path

from sign_semantics.human_associations import aggregate_model_tokens_by_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate_model_tokens_by_code(args.tokens, args.manifest, args.output)


if __name__ == "__main__":
    main()
