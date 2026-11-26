from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

from web_overlay.semantic_catalog import (
    SemanticCatalogValidationError,
    compile_semantic_catalog,
    load_semantic_catalog,
    publish_semantic_catalog,
)
from web_overlay.semantic_ontology import OntologyValidationError, load_ontology


FUNCTIONS = (
    "emphasize",
    "enumerate",
    "contrast_compare",
    "refer_self",
    "refer_other",
    "refer_external",
    "depict_size_shape",
    "depict_direction_motion",
    "turn_open",
    "turn_yield",
    "greet_farewell",
    "thank",
    "affirm",
    "deny",
    "certainty",
    "uncertainty",
    "enthusiasm",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atom_hash(atom: dict[str, object]) -> str:
    payload = deepcopy(atom)
    payload["identity"].pop("content_hash", None)
    return _sha(_canonical_bytes(payload))


class _SemanticFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run = root / "source_run"
        self.run.mkdir()
        self.ontology_path = root / "dialogue_function_17_v1.json"
        self.final_path = root / "final_labels.jsonl"
        self.removed_path = root / "removed_actions.jsonl"
        self.inventory_path = root / "approved_inventory.json"
        self.sidecar_path = root / "exact_motion_sides_v1.json"
        self.neutral_path = root / "safe_neutral.json"
        self.inventory_rows: list[dict[str, object]] = []
        self.final_rows: list[dict[str, object]] = []
        self.removed_rows: list[dict[str, object]] = []
        self.side_rows: list[dict[str, object]] = []
        self._write_ontology()
        neutral = {
            "format": "g1_joint_motion_v1",
            "name": "safe_neutral",
            "duration": 0.8,
            "joints": ["left_elbow", "right_elbow"],
            "frames": [
                {"t": 0.0, "q": {"left_elbow": 0.1, "right_elbow": 0.1}},
                {"t": 0.8, "q": {"left_elbow": 0.1, "right_elbow": 0.1}},
            ],
        }
        self.neutral_path.write_bytes(_canonical_bytes(neutral) + b"\n")

    def _write_ontology(self) -> None:
        categories = {
            "emphasize": "emphasis",
            "enumerate": "enumeration",
            "contrast_compare": "contrast",
            "refer_self": "reference",
            "refer_other": "reference",
            "refer_external": "reference",
            "depict_size_shape": "depiction",
            "depict_direction_motion": "depiction",
            "turn_open": "turn_management",
            "turn_yield": "turn_management",
            "greet_farewell": "social",
            "thank": "social",
            "affirm": "social",
            "deny": "social",
            "certainty": "stance",
            "uncertainty": "stance",
            "enthusiasm": "stance",
        }
        ontology = {
            "schema_version": "dialogue_function_17_v1",
            "label_scope": "planner_function",
            "derived_from": "f18_codebook_v1",
            "excluded_functions": [
                {
                    "function": "present_explain",
                    "reason": "No retained Approved examples in the formal review snapshot.",
                }
            ],
            "functions": [
                {
                    "function": name,
                    "planner_category": categories[name],
                    "common_families": ["beat"],
                    "definition_zh": f"fixture definition for {name}",
                }
                for name in FUNCTIONS
            ],
            "separation_rule": "BEAT family is evidence, never a planner label.",
        }
        self.ontology_path.write_bytes(_canonical_bytes(ontology) + b"\n")

    def add(self, index: int, *, removed: bool = False, label: str = "emphasize") -> None:
        atom_id = f"beat2.fixture.s0.1.f{index * 10}-{index * 10 + 2}.atom{index}"
        trajectory = [
            {
                "t": 0.0,
                "q": {"left_elbow": 0.0, "right_elbow": 0.0},
            },
            {
                "t": 0.2,
                "q": {"left_elbow": 0.2, "right_elbow": 0.02},
            },
        ]
        atom: dict[str, object] = {
            "schema_version": "robot.gesture_atom.v1",
            "identity": {"id": atom_id, "revision": 1},
            "semantics": {"handedness": "left"},
            "source": {
                "dataset": "BEAT2-retargeted-G1",
                "speaker": "fixture",
                "session": "0",
                "clip": f"fixture_0_{index}",
                "relative_path": f"fixture_{index}.pkl",
                "input_sha256": str(index) * 64,
                "frame_start": index * 10,
                "frame_end": index * 10 + 2,
            },
            "timing": {"nominal_duration_s": 0.2},
            "control": {
                "joints": ["left_elbow", "right_elbow"],
                "trajectory": trajectory,
            },
            "safety": {
                "metrics": {
                    "passed": True,
                    "base_mode": "fixed",
                    "kinematics_passed": True,
                    "position_limits_checked": True,
                    "collision_checked": True,
                },
                "collision": {"checked": True, "passed": True},
                "violations": [],
                "certified_backends": ["mujoco-g1-29dof"],
            },
        }
        content_hash = _atom_hash(atom)
        atom["identity"]["content_hash"] = content_hash
        atom_relative = f"candidates/{atom_id}.json"
        preview_relative = f"previews/{atom_id}.json"
        report_relative = f"reports/{atom_id}.json"
        atom_path = self.run / atom_relative
        preview_path = self.run / preview_relative
        report_path = self.run / report_relative
        atom_path.parent.mkdir(exist_ok=True)
        preview_path.parent.mkdir(exist_ok=True)
        report_path.parent.mkdir(exist_ok=True)
        atom_path.write_bytes(json.dumps(atom, indent=2).encode() + b"\n")
        preview = {
            "format": "g1_joint_motion_v1",
            "name": f"{atom_id}.r1",
            "duration": 0.2,
            "joints": ["left_elbow", "right_elbow"],
            "interpolation": "linear",
            "frames": deepcopy(trajectory),
            "source_atom_hash": content_hash,
            "parent_atom_hash": content_hash,
        }
        preview_path.write_bytes(json.dumps(preview, indent=2).encode() + b"\n")
        report = {
            "result": "pass",
            "eligible_for_backend_certification": True,
            "certified_backends": ["mujoco-g1-29dof"],
            "collision": {"result": "pass"},
        }
        report_path.write_bytes(json.dumps(report, indent=2).encode() + b"\n")
        exact = {
            "atom_id": atom_id,
            "version": "candidate",
            "revision": 1,
            "content_sha256": content_hash,
        }
        review = {
            "decision": "remove_atom" if removed else "approve_qwen",
            "event_id": f"human_review_{index}",
            "qwen_basis": "fixture",
            "qwen_label": "emphasize",
            "review_note": "",
            "reviewed_at_utc": "2026-08-29T00:00:00Z",
            "reviewer": "operator",
        }
        if removed:
            self.removed_rows.append(
                {"schema_version": "f18_removed_action_v1", "index": index, **exact, "human_review": review}
            )
        else:
            self.final_rows.append(
                {
                    "schema_version": "f18_human_label_v1",
                    "index": index,
                    **exact,
                    "label": label,
                    "human_review": review,
                }
            )
        self.inventory_rows.append(
            {
                "index": index,
                **exact,
                "approved": True,
                "approval_event_id": f"approval_{index}",
                "decision_source_event_id": f"approval_{index}",
                "latest_annotation_event_id": f"approval_{index}",
                "approval_timestamp": "2026-08-24T00:00:00Z",
                "fixed_base_pass": True,
                "variant": "raw",
                "atom_file": atom_relative,
                "atom_file_sha256": _sha(atom_path.read_bytes()),
                "preview_file": preview_relative,
                "preview_file_sha256": _sha(preview_path.read_bytes()),
                "report_file": report_relative,
                "report_file_sha256": _sha(report_path.read_bytes()),
                "clip_file": f"clips/{index:03d}.mp4",
                "clip_sha256": "a" * 64,
                "duration_s": 0.2,
                "source": deepcopy(atom["source"]),
                "source_window": {
                    "source_fps": 30.0,
                    "frame_start": index * 10,
                    "frame_end_inclusive": index * 10 + 2,
                    "frame_end_semantics": "inclusive",
                    "time_start_s": 0.0,
                    "time_end_inclusive_s": 0.2,
                    "coverage_end_exclusive_s": 0.233333333333,
                },
                "semantics": {"handedness": "left"},
            }
        )
        self.side_rows.append(
            {
                "index": index,
                **exact,
                "approval_event_id": f"approval_{index}",
                "atom_file_sha256": _sha(atom_path.read_bytes()),
                "variant": "raw",
                "stored_handedness": "left",
                "stored_left_path_rad": 0.2,
                "stored_right_path_rad": 0.02,
                "exact_trajectory_handedness": "left",
                "left_path_rad": 0.2,
                "right_path_rad": 0.02,
                "left_share": 0.2 / 0.22,
                "balance_ratio": 0.1,
                "handedness_differs_from_stored": False,
                "motion_totals_differ_from_stored": False,
            }
        )

    def write(self) -> None:
        self.final_path.write_bytes(
            b"".join(_canonical_bytes(row) + b"\n" for row in self.final_rows)
        )
        self.removed_path.write_bytes(
            b"".join(_canonical_bytes(row) + b"\n" for row in self.removed_rows)
        )
        inventory = {
            "schema_version": "approved_atom_inventory_v1",
            "catalog_id": "fixture_catalog",
            "source_run_dir": str(self.run),
            "review_export_dir": str(self.root / "review_export"),
            "annotation_snapshot": {"sha256": "b" * 64},
            "run_manifest": {"sha256": "c" * 64},
            "review_manifest": {"sha256": "d" * 64},
            "approved_count": len(self.inventory_rows),
            "atoms": self.inventory_rows,
        }
        inventory_raw = _canonical_bytes(inventory) + b"\n"
        self.inventory_path.write_bytes(inventory_raw)
        sidecar = {
            "schema_version": "f18_exact_motion_sides_v1",
            "catalog_id": "fixture_catalog",
            "item_count": len(self.side_rows),
            "dataset_manifest_sha256": "e" * 64,
            "approved_inventory_sha256": _sha(inventory_raw),
            "algorithm": {
                "algorithm_id": "g1_joint_total_variation_handedness_v1",
                "definition": "sum(abs(delta_q)) per joint, summed by robot left_/right_ joint prefix",
                "unit": "aggregate_joint_radians",
                "bilateral_min_ratio": 0.55,
                "none_max_fraction": 0.15,
                "coordinate_frame": "robot_anatomical_joint_names",
            },
            "items": self.side_rows,
        }
        self.sidecar_path.write_bytes(_canonical_bytes(sidecar) + b"\n")

    def compile(self):
        self.write()
        return compile_semantic_catalog(
            ontology_path=self.ontology_path,
            final_labels_path=self.final_path,
            removed_actions_path=self.removed_path,
            approved_inventory_path=self.inventory_path,
            exact_sidecar_path=self.sidecar_path,
            neutral_motion_path=self.neutral_path,
        )


class SemanticCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.fixture = _SemanticFixture(self.root)
        self.fixture.add(1, label="emphasize")
        self.fixture.add(2, removed=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_compiles_planner_rows_and_immutable_inline_motion(self) -> None:
        catalog = self.fixture.compile()

        self.assertEqual(catalog.schema_version, "semantic_atom_catalog_v1")
        self.assertEqual(catalog.ontology_version, "dialogue_function_17_v1")
        self.assertEqual(catalog.retained_count, 1)
        self.assertEqual(catalog.removed_count, 1)
        self.assertEqual(catalog.neutral_q["left_elbow"], 0.1)
        self.assertEqual(catalog.neutral_duration_us, 800_000)
        self.assertEqual(catalog.neutral_motion["interpolation"], "linear")
        self.assertEqual(catalog.neutral_motion["frames"][-1]["t"], 0.8)
        self.assertIsInstance(catalog.neutral_motion, MappingProxyType)
        self.assertEqual(len(catalog.rows), 1)
        row = catalog.rows[0]
        self.assertEqual(row.function, "emphasize")
        self.assertEqual(row.duration_us, 200_000)
        self.assertEqual(row.execution_side, "left")
        self.assertTrue(row.approved)
        self.assertFalse(row.removed)
        self.assertTrue(row.source_id.startswith("BEAT2-retargeted-G1:"))
        self.assertIsInstance(row.motion, MappingProxyType)
        self.assertEqual(row.motion["source_atom_hash"], row.content_sha256)
        self.assertEqual(row.motion["frames"][1]["q"]["left_elbow"], 0.2)
        self.assertEqual(row.entry_q["left_elbow"], 0.0)
        self.assertEqual(row.exit_q["left_elbow"], 0.2)
        with self.assertRaises(TypeError):
            row.motion["duration"] = 1.0
        self.assertEqual(catalog.by_function["emphasize"], (row.entry_id,))
        self.assertEqual(tuple(catalog.by_function), FUNCTIONS)
        self.assertNotIn("present_explain", catalog.by_function)

    def test_content_addressed_publish_is_idempotent_and_loads(self) -> None:
        catalog = self.fixture.compile()
        output = self.root / "published"
        first = publish_semantic_catalog(catalog, output)
        second = publish_semantic_catalog(catalog, output)
        self.assertEqual(first, second)
        self.assertEqual(first.name, f"semantic_atom_catalog_v1.{catalog.content_sha256}.json")
        self.assertEqual(_sha(first.read_bytes()), catalog.content_sha256)

        loaded = load_semantic_catalog(first)
        self.assertEqual(loaded.content_sha256, catalog.content_sha256)
        self.assertEqual(loaded.rows[0].entry_id, catalog.rows[0].entry_id)
        self.assertIsInstance(loaded.rows[0].motion, MappingProxyType)
        self.assertEqual(loaded.neutral_duration_us, 800_000)
        self.assertEqual(loaded.neutral_motion["frames"][-1]["q"], loaded.neutral_q)
        with self.assertRaises(TypeError):
            loaded.neutral_motion["duration"] = 2.0

    def test_ontology_is_exactly_17_and_excludes_present_explain(self) -> None:
        ontology = load_ontology(self.fixture.ontology_path)
        self.assertEqual(ontology.function_names, FUNCTIONS)
        self.assertEqual(ontology.excluded_functions, ("present_explain",))

        payload = json.loads(self.fixture.ontology_path.read_text())
        payload["functions"][0] = {
            "function": "present_explain",
            "planner_category": "presentation",
            "common_families": ["metaphoric"],
            "definition_zh": "forbidden",
        }
        self.fixture.ontology_path.write_bytes(_canonical_bytes(payload) + b"\n")
        with self.assertRaisesRegex(OntologyValidationError, "present_explain"):
            load_ontology(self.fixture.ontology_path)

    def test_rejects_present_explain_and_incomplete_review_partition(self) -> None:
        self.fixture.final_rows[0]["label"] = "present_explain"
        with self.assertRaisesRegex(SemanticCatalogValidationError, "present_explain"):
            self.fixture.compile()

        self.fixture.final_rows[0]["label"] = "emphasize"
        self.fixture.removed_rows.clear()
        with self.assertRaisesRegex(SemanticCatalogValidationError, "partition"):
            self.fixture.compile()

    def test_rejects_tampered_atom_and_non_pass_report(self) -> None:
        row = self.fixture.inventory_rows[0]
        atom_path = self.fixture.run / row["atom_file"]
        atom = json.loads(atom_path.read_text())
        atom["control"]["trajectory"][1]["q"]["left_elbow"] = 0.3
        atom_path.write_bytes(json.dumps(atom, indent=2).encode() + b"\n")
        row["atom_file_sha256"] = _sha(atom_path.read_bytes())
        self.fixture.side_rows[0]["atom_file_sha256"] = row["atom_file_sha256"]
        with self.assertRaisesRegex(SemanticCatalogValidationError, "content hash"):
            self.fixture.compile()

        second_root = self.root / "report_case"
        second_root.mkdir()
        second = _SemanticFixture(second_root)
        second.add(1, label="emphasize")
        second.add(2, removed=True)
        row = second.inventory_rows[0]
        report_path = second.run / row["report_file"]
        report = json.loads(report_path.read_text())
        report["result"] = "fail"
        report_path.write_bytes(json.dumps(report, indent=2).encode() + b"\n")
        row["report_file_sha256"] = _sha(report_path.read_bytes())
        with self.assertRaisesRegex(SemanticCatalogValidationError, "fixed-base PASS"):
            second.compile()

    def test_rejects_preview_motion_not_exactly_equal_to_atom(self) -> None:
        row = self.fixture.inventory_rows[0]
        preview_path = self.fixture.run / row["preview_file"]
        preview = json.loads(preview_path.read_text())
        preview["frames"][1]["q"]["left_elbow"] = 0.25
        preview_path.write_bytes(json.dumps(preview, indent=2).encode() + b"\n")
        row["preview_file_sha256"] = _sha(preview_path.read_bytes())
        with self.assertRaisesRegex(SemanticCatalogValidationError, "preview.*trajectory"):
            self.fixture.compile()

    def test_raw_preview_may_omit_parent_hash_and_is_normalized(self) -> None:
        row = self.fixture.inventory_rows[0]
        preview_path = self.fixture.run / row["preview_file"]
        preview = json.loads(preview_path.read_text())
        preview.pop("parent_atom_hash")
        preview_path.write_bytes(json.dumps(preview, indent=2).encode() + b"\n")
        row["preview_file_sha256"] = _sha(preview_path.read_bytes())

        catalog = self.fixture.compile()

        self.assertIsNone(catalog.rows[0].motion["parent_atom_hash"])

    def test_rejects_sidecar_not_bound_to_exact_trajectory(self) -> None:
        self.fixture.side_rows[0]["left_path_rad"] = 99.0
        with self.assertRaisesRegex(SemanticCatalogValidationError, "left_path_rad"):
            self.fixture.compile()


if __name__ == "__main__":
    unittest.main()
