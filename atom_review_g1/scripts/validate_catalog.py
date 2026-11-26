#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from g1_atom_player.catalog import load_catalog  # noqa: E402
from g1_atom_player.config import load_config  # noqa: E402
from g1_atom_player.server import _strict_verify  # noqa: E402
from g1_atom_player.validation import load_limits, validate_atom  # noqa: E402


def main() -> int:
    config = load_config(ROOT)
    catalog = load_catalog(config.catalog_path)
    _strict_verify(config, catalog)
    limits = load_limits(config.limits_path)
    results = [validate_atom(atom, limits) for atom in catalog.entries]
    passed = sum(bool(row["ok"]) for row in results)
    payload = {
        "ok": True,
        "catalog_sha256": catalog.file_sha256,
        "atom_count": len(results),
        "hardware_gate_passed": passed,
        "hardware_gate_blocked": len(results) - passed,
        "note": "A block never modifies the Atom; it excludes that exact version from real execution.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
