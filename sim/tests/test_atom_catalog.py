from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from web_overlay.atom_catalog import CatalogValidationError, load_catalog
from web_overlay.random_planner import RandomSequencePlanner


REPAIR_VERSION_1 = "1" * 32
REPAIR_VERSION_2 = "2" * 32


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _atom_hash(atom: dict[str, object]) -> str:
    payload = deepcopy(atom)
    payload["identity"].pop("content_hash", None)
    return _canonical_sha256(payload)


class _CatalogFixture:
    def __init__(self, root: Path) -> None:
        self.run = root / "source_run"
        self.run.mkdir(parents=True)
        (self.run / "curation").mkdir()
        self.entries: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self.manifest_atoms: list[dict[str, object]] = []

    @staticmethod
    def _atom(atom_id: str, revision: int, *, passed: bool) -> dict[str, object]:
        atom: dict[str, object] = {
            "schema_version": "robot.gesture_atom.v1",
            "identity": {"id": atom_id, "revision": revision},
            "control": {
                "joints": ["left_elbow", "right_elbow"],
                "trajectory": [
                    {
                        "t": 0.0,
                        "q": {"left_elbow": 0.1, "right_elbow": -0.1},
                    },
                    {
                        "t": 0.2,
                        "q": {"left_elbow": 0.2, "right_elbow": -0.2},
                    },
                ],
            },
            "safety": {
                "metrics": {
                    "passed": passed,
                    "base_mode": "fixed",
                    "kinematics_passed": passed,
                    "position_limits_checked": True,
                    "collision_checked": True,
                },
                "collision": {"checked": True, "passed": passed},
                "violations": [] if passed else [{"code": "fixture_failure"}],
                "certified_backends": ["mujoco-g1-29dof"],
            },
        }
        atom["identity"]["content_hash"] = _atom_hash(atom)
        return atom

    def add(
        self,
        atom_id: str,
        *,
        version: str,
        revision: int = 2,
        approved: object = True,
        decision: str = "approve",
        passed: bool = True,
    ) -> Path:
        atom = self._atom(atom_id, revision, passed=passed)
        content_hash = atom["identity"]["content_hash"]
        if version == "candidate":
            atom_path = self.run / "candidates" / f"{atom_id}.json"
            report_path = self.run / "reports" / f"{atom_id}.motion_report.json"
        else:
            bundle = (
                self.run
                / "curation"
                / "repaired"
                / atom_id
                / f"r{revision}.{content_hash[:12]}.{version}"
            )
            atom_path = bundle / "atom.json"
            report_path = bundle / "motion_report.json"
        atom_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atom_path.write_text(json.dumps(atom, indent=2) + "\n", encoding="utf-8")

        report = {
            "result": "pass" if passed else "fail",
            "collision": {"result": "pass" if passed else "fail"},
            "eligible_for_backend_certification": passed,
            "certified_backends": ["mujoco-g1-29dof"],
        }
        if version == "candidate":
            report.update({"atom_id": atom_id, "atom_content_hash": content_hash})
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        if version == "candidate":
            self.manifest_atoms.append(
                {
                    "id": atom_id,
                    "content_hash": content_hash,
                    "atom": atom_path.relative_to(self.run).as_posix(),
                    "report": report_path.relative_to(self.run).as_posix(),
                    "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    "constraint_passed": passed,
                    "collision_checked": True,
                }
            )
        else:
            self.events.append(
                {
                    "event_id": version,
                    "event_type": "repair",
                    "atom_id": atom_id,
                    "revision": revision,
                    "content_hash": content_hash,
                    "atom": atom_path.relative_to(self.run).as_posix(),
                    "report": report_path.relative_to(self.run).as_posix(),
                    "report_sha256": _canonical_sha256(report),
                    "auto_repair": {"status": "repaired"},
                }
            )
        self.events.append(
            {
                "event_id": f"review_{atom_id}",
                "event_type": "annotation",
                "atom_id": atom_id,
                "reviewed_version": version,
                "reviewed_revision": revision,
                "reviewed_content_hash": content_hash,
                "decision": decision,
            }
        )
        self.entries.append(
            {
                "atom_id": atom_id,
                "version": version,
                "revision": revision,
                "content_sha256": content_hash,
                "approved": approved,
                "primary_category": "emphasis",
                "tags": ["bimanual"],
                "label_source": "manual_seed",
            }
        )
        return atom_path

    def write(self) -> Path:
        (self.run / "run_manifest.json").write_text(
            json.dumps({"atoms": self.manifest_atoms}, indent=2) + "\n",
            encoding="utf-8",
        )
        annotations = self.run / "curation" / "annotations.jsonl"
        annotations.write_text(
            "".join(json.dumps(event) + "\n" for event in self.events),
            encoding="utf-8",
        )
        catalog = self.run.parent / "atom_catalog_v1.json"
        catalog.write_text(
            json.dumps(
                {
                    "schema_version": "atom_catalog_v1",
                    "source_run_dir": str(self.run),
                    "atoms": self.entries,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return catalog


class AtomCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_loads_exact_approved_repair_and_candidate(self) -> None:
        fixture = _CatalogFixture(self.root)
        fixture.add("atom_001", version=REPAIR_VERSION_1)
        fixture.add("atom_002", version="candidate", revision=1)

        catalog = load_catalog(fixture.write())

        self.assertEqual([item.atom_id for item in catalog.approved_atoms()], ["atom_001", "atom_002"])
        repaired, candidate = catalog.approved_atoms()
        self.assertEqual(repaired.version, REPAIR_VERSION_1)
        self.assertEqual(repaired.first_q, {"left_elbow": 0.1, "right_elbow": -0.1})
        self.assertEqual(repaired.atom_path.name, "atom.json")
        self.assertEqual(candidate.version, "candidate")
        self.assertTrue(catalog.annotations_sha256)

    def test_rejects_non_literal_true_and_duplicate_atom_id(self) -> None:
        fixture = _CatalogFixture(self.root)
        fixture.add("atom_001", version=REPAIR_VERSION_1, approved=1)
        with self.assertRaisesRegex(CatalogValidationError, "approved.*literal true"):
            load_catalog(fixture.write())

        fixture.entries[0]["approved"] = True
        fixture.entries.append(deepcopy(fixture.entries[0]))
        with self.assertRaisesRegex(CatalogValidationError, "duplicate atom_id"):
            load_catalog(fixture.write())

    def test_rejects_schema_drift_and_unknown_fields(self) -> None:
        fixture = _CatalogFixture(self.root)
        fixture.add("atom_001", version=REPAIR_VERSION_1)
        path = fixture.write()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = "atom_catalog_v2"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CatalogValidationError, "schema_version"):
            load_catalog(path)

        payload["schema_version"] = "atom_catalog_v1"
        payload["surprise"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CatalogValidationError, "unknown field"):
            load_catalog(path)

    def test_rejects_tampered_content_and_missing_file(self) -> None:
        fixture = _CatalogFixture(self.root)
        atom_path = fixture.add("atom_001", version=REPAIR_VERSION_1)
        catalog_path = fixture.write()
        atom = json.loads(atom_path.read_text(encoding="utf-8"))
        atom["control"]["trajectory"][0]["q"]["left_elbow"] = 9.0
        atom_path.write_text(json.dumps(atom), encoding="utf-8")
        with self.assertRaisesRegex(CatalogValidationError, "content hash"):
            load_catalog(catalog_path)

        atom_path.unlink()
        with self.assertRaisesRegex(CatalogValidationError, "does not exist"):
            load_catalog(catalog_path)

    def test_rejects_empty_trajectory_even_with_self_consistent_binding(self) -> None:
        fixture = _CatalogFixture(self.root)
        atom_path = fixture.add("atom_001", version="candidate", revision=1)
        catalog_path = fixture.write()
        atom = json.loads(atom_path.read_text(encoding="utf-8"))
        atom["control"]["trajectory"] = []
        atom["identity"]["content_hash"] = _atom_hash(atom)
        fixture.entries[0]["content_sha256"] = atom["identity"]["content_hash"]
        fixture.events[0]["reviewed_content_hash"] = atom["identity"]["content_hash"]
        fixture.manifest_atoms[0]["content_hash"] = atom["identity"]["content_hash"]
        atom_path.write_text(json.dumps(atom), encoding="utf-8")

        report_path = fixture.run / fixture.manifest_atoms[0]["report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["atom_content_hash"] = atom["identity"]["content_hash"]
        report_path.write_text(json.dumps(report), encoding="utf-8")
        fixture.manifest_atoms[0]["report_sha256"] = hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
        fixture.write()
        with self.assertRaisesRegex(CatalogValidationError, "trajectory"):
            load_catalog(catalog_path)

    def test_rejects_non_pass_and_non_current_exact_approval(self) -> None:
        fixture = _CatalogFixture(self.root)
        fixture.add("atom_001", version=REPAIR_VERSION_1, passed=False)
        with self.assertRaisesRegex(CatalogValidationError, "fixed-base PASS"):
            load_catalog(fixture.write())

        fixture = _CatalogFixture(self.root / "second")
        fixture.add("atom_002", version=REPAIR_VERSION_2, decision="reject")
        with self.assertRaisesRegex(CatalogValidationError, "not currently approved"):
            load_catalog(fixture.write())

    def test_rejects_missing_exact_repair_event(self) -> None:
        fixture = _CatalogFixture(self.root)
        fixture.add("atom_001", version=REPAIR_VERSION_1)
        fixture.events = [event for event in fixture.events if event["event_type"] != "repair"]
        with self.assertRaisesRegex(CatalogValidationError, "repair event"):
            load_catalog(fixture.write())

    def test_rejects_symlinked_evidence_files_and_path_components(self) -> None:
        evidence_paths = (
            "run_manifest.json",
            "curation/annotations.jsonl",
            "candidates/atom_001.json",
            "reports/atom_001.motion_report.json",
        )
        for case_index, relative in enumerate(evidence_paths):
            with self.subTest(relative=relative):
                case_root = self.root / f"evidence_{case_index}"
                fixture = _CatalogFixture(case_root)
                fixture.add("atom_001", version="candidate", revision=1)
                catalog_path = fixture.write()
                evidence_path = fixture.run / relative
                outside = case_root / f"outside_{case_index}.json"
                evidence_path.replace(outside)
                evidence_path.symlink_to(outside)
                with self.assertRaisesRegex(CatalogValidationError, "unsafe evidence path"):
                    load_catalog(catalog_path)

        case_root = self.root / "evidence_parent"
        fixture = _CatalogFixture(case_root)
        fixture.add("atom_001", version="candidate", revision=1)
        catalog_path = fixture.write()
        candidate_dir = fixture.run / "candidates"
        outside_dir = case_root / "outside_candidates"
        shutil.move(candidate_dir, outside_dir)
        candidate_dir.symlink_to(outside_dir, target_is_directory=True)
        with self.assertRaisesRegex(CatalogValidationError, "unsafe evidence path"):
            load_catalog(catalog_path)

    def test_loaded_atom_payload_is_recursively_immutable(self) -> None:
        fixture = _CatalogFixture(self.root)
        fixture.add("atom_001", version="candidate", revision=1)
        atom = load_catalog(fixture.write()).atoms[0]
        before = atom.first_q

        with self.assertRaises(TypeError):
            atom.atom_payload["control"]["trajectory"][0]["q"]["left_elbow"] = 123.0

        self.assertEqual(atom.first_q, before)

    def test_rejects_unhashable_atom_id_as_catalog_validation_error(self) -> None:
        fixture = _CatalogFixture(self.root)
        fixture.add("atom_001", version="candidate", revision=1)
        path = fixture.write()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["atoms"][0]["atom_id"] = ["not", "hashable"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(CatalogValidationError, "atom_id"):
            load_catalog(path)

    def test_rejects_noncanonical_versions(self) -> None:
        for case_index, version in enumerate(
            ("repair001", "A" * 32, "a" * 31, "a" * 33)
        ):
            with self.subTest(version=version):
                fixture = _CatalogFixture(self.root / f"version_{case_index}")
                fixture.add("atom_001", version=version)
                with self.assertRaisesRegex(CatalogValidationError, "version"):
                    load_catalog(fixture.write())

    def test_rejects_wrong_atom_schema_even_with_self_consistent_binding(self) -> None:
        fixture = _CatalogFixture(self.root)
        atom_path = fixture.add("atom_001", version="candidate", revision=1)
        catalog_path = fixture.write()
        atom = json.loads(atom_path.read_text(encoding="utf-8"))
        atom["schema_version"] = "robot.gesture_atom.v2"
        atom["identity"]["content_hash"] = _atom_hash(atom)
        content_hash = atom["identity"]["content_hash"]
        fixture.entries[0]["content_sha256"] = content_hash
        fixture.events[0]["reviewed_content_hash"] = content_hash
        fixture.manifest_atoms[0]["content_hash"] = content_hash
        atom_path.write_text(json.dumps(atom), encoding="utf-8")

        report_path = fixture.run / fixture.manifest_atoms[0]["report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["atom_content_hash"] = content_hash
        report_path.write_text(json.dumps(report), encoding="utf-8")
        fixture.manifest_atoms[0]["report_sha256"] = hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
        fixture.write()
        with self.assertRaisesRegex(CatalogValidationError, "schema_version"):
            load_catalog(catalog_path)


class RandomSequencePlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        fixture = _CatalogFixture(root)
        for index in range(1, 6):
            fixture.add(f"atom_{index:03d}", version=f"{index:032x}")
        self.catalog = load_catalog(fixture.write())

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_fixed_seed_is_reproducible_without_replacement(self) -> None:
        planner = RandomSequencePlanner()

        first = planner.plan(turn_id="turn_001", atom_catalog=self.catalog, count=3, seed=42)
        second = planner.plan(turn_id="turn_001", atom_catalog=self.catalog, count=3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual([item.atom_id for item in first.atoms], ["atom_001", "atom_005", "atom_003"])
        self.assertEqual(len({item.atom_id for item in first.atoms}), 3)
        self.assertEqual(
            first.to_dict(),
            {
                "turn_id": "turn_001",
                "strategy": "random_v0",
                "seed": 42,
                "atoms": [
                    {"atom_id": "atom_001", "revision": 2},
                    {"atom_id": "atom_005", "revision": 2},
                    {"atom_id": "atom_003", "revision": 2},
                ],
            },
        )
        log_record = first.to_log_record()
        self.assertEqual(log_record["atoms"][0]["version"], f"{1:032x}")
        self.assertEqual(log_record["atoms"][0]["content_sha256"], first.atoms[0].content_sha256)

    def test_rejects_invalid_count_turn_id_and_seed(self) -> None:
        planner = RandomSequencePlanner()
        with self.assertRaisesRegex(ValueError, "count"):
            planner.plan(turn_id="turn_001", atom_catalog=self.catalog, count=6, seed=42)
        with self.assertRaisesRegex(ValueError, "turn_id"):
            planner.plan(turn_id="", atom_catalog=self.catalog, count=3, seed=42)
        with self.assertRaisesRegex(ValueError, "seed"):
            planner.plan(turn_id="turn_001", atom_catalog=self.catalog, count=3, seed=True)


if __name__ == "__main__":
    unittest.main()
