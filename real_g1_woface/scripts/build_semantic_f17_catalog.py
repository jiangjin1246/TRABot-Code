#!/usr/bin/env python3
"""Compile the formal F17 semantic Atom catalog after exact validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_overlay.semantic_catalog import (  # noqa: E402
    SemanticCatalogValidationError,
    compile_semantic_catalog,
    publish_semantic_catalog,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable, content-addressed dialogue_function_17_v1 "
            "catalog from the formal human-review snapshot."
        )
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=PROJECT_ROOT / "ontology" / "dialogue_function_17_v1.json",
        help="frozen dialogue_function_17_v1 ontology JSON",
    )
    parser.add_argument("--final-labels", type=Path, required=True)
    parser.add_argument("--removed-actions", type=Path, required=True)
    parser.add_argument("--approved-inventory", type=Path, required=True)
    parser.add_argument("--exact-sidecar", type=Path, required=True)
    parser.add_argument(
        "--neutral-motion",
        type=Path,
        default=PROJECT_ROOT / "motions" / "safe_neutral.json",
        help="canonical transition neutral motion (default: motions/safe_neutral.json)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-retained", type=int, default=113)
    parser.add_argument("--expected-removed", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalog = compile_semantic_catalog(
            ontology_path=args.ontology,
            final_labels_path=args.final_labels,
            removed_actions_path=args.removed_actions,
            approved_inventory_path=args.approved_inventory,
            exact_sidecar_path=args.exact_sidecar,
            neutral_motion_path=args.neutral_motion,
            expected_retained=args.expected_retained,
            expected_removed=args.expected_removed,
        )
        output = publish_semantic_catalog(catalog, args.output_dir)
    except SemanticCatalogValidationError as exc:
        print(f"semantic catalog validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": catalog.schema_version,
                "ontology_version": catalog.ontology_version,
                "path": str(output),
                "sha256": catalog.content_sha256,
                "source_count": catalog.source_count,
                "retained_count": catalog.retained_count,
                "removed_count": catalog.removed_count,
                "function_counts": {
                    function: len(entry_ids)
                    for function, entry_ids in catalog.by_function.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
