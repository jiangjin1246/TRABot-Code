from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from g1_atom_player.catalog import (
    CatalogError,
    EXPECTED_JOINTS,
    canonical_sha256,
    load_catalog,
    minimum_jerk_duration,
)
from g1_atom_player.reviews import ReviewStore
from g1_atom_player.server import ApiHandler, AtomPlayerApp, AuditLog, PreviewAsset, RunManager
from g1_atom_player.validation import LimitsError, load_limits, validate_atom


def motion(last: float = 0.1, delta: float = 0.1) -> dict:
    q0 = {joint: 0.0 for joint in EXPECTED_JOINTS}
    q1 = {joint: delta for joint in EXPECTED_JOINTS}
    return {
        "format": "g1_joint_motion_v1",
        "name": "fixture",
        "joints": list(EXPECTED_JOINTS),
        "frames": [{"t": 0.0, "q": q0}, {"t": last, "q": q1}],
        "duration": last,
        "interpolation": "linear",
    }


def catalog_payload() -> dict:
    atom_motion = motion()
    neutral_motion = motion(last=0.8, delta=0.0)
    return {
        "neutral": {
            "q": {joint: 0.0 for joint in EXPECTED_JOINTS},
            "motion": neutral_motion,
        },
        "entries": [
            {
                "index": 1,
                "entry_id": "a" * 64,
                "atom_id": "fixture.atom",
                "function": "emphasize",
                "version": "candidate",
                "revision": 1,
                "content_sha256": "b" * 64,
                "duration_us": 100000,
                "motion": atom_motion,
            }
        ],
    }


def sequence_catalog_payload(count: int = 3) -> dict:
    payload = catalog_payload()
    entries = []
    for offset in range(count):
        item = json.loads(json.dumps(payload["entries"][0]))
        item["index"] = offset + 1
        item["entry_id"] = chr(ord("a") + offset) * 64
        item["atom_id"] = f"fixture.atom.{offset + 1}"
        item["content_sha256"] = chr(ord("b") + offset) * 64
        item["motion"] = motion(delta=0.1 * (offset + 1))
        entries.append(item)
    payload["entries"] = entries
    return payload


class RecordingRuns:
    def __init__(self) -> None:
        self.sequences: list[tuple] = []

    def start_sequence(self, atoms: tuple) -> dict:
        self.sequences.append(atoms)
        return {
            "status": "running",
            "mode": "sequence",
            "entry_ids": [atom.entry_id for atom in atoms],
        }


class CatalogTests(unittest.TestCase):
    def write_catalog(self, directory: Path, payload: dict) -> Path:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        digest = hashlib.sha256(body).hexdigest()
        path = directory / f"semantic_atom_catalog_v1.{digest}.json"
        path.write_bytes(body)
        return path

    def make_sequence_app(self, directory: Path) -> AtomPlayerApp:
        catalog = load_catalog(self.write_catalog(directory, sequence_catalog_payload()))
        app = object.__new__(AtomPlayerApp)
        app.catalog = catalog
        app.limits = {
            "source": "fixture",
            "position_margin_rad": 0.0,
            "joints": {
                joint: {"min_rad": -2.0, "max_rad": 2.0}
                for joint in EXPECTED_JOINTS
            },
        }
        preview_sha256_by_atom: dict[str, str] = {}
        app.previews = {}
        for atom in catalog.entries:
            body = f"reviewed preview for {atom.atom_id}".encode()
            path = directory / f"{atom.index}.mp4"
            path.write_bytes(body)
            digest = hashlib.sha256(body).hexdigest()
            app.previews[atom.atom_id] = PreviewAsset(path=path, sha256=digest)
            preview_sha256_by_atom[atom.atom_id] = digest
        app.reviews = ReviewStore(
            directory / "reviews.json", catalog, preview_sha256_by_atom
        )
        for atom in catalog.entries:
            app.reviews.set(atom, status="approved", reviewer="tester")
        runner = directory / "runner"
        runner.write_text("fixture runner\n", encoding="utf-8")
        runner.chmod(0o755)
        app.config = SimpleNamespace(
            real_enabled=True,
            runner_path=runner,
            catalog_path=catalog.path,
            limits_path=directory / "limits.json",
            network_interface="eno2",
            port=0,
        )
        app.csrf_token = "fixture-token"
        app.audit = AuditLog(directory / "audit.jsonl")
        app.runs = RecordingRuns()
        return app

    def post_json(self, app: AtomPlayerApp, path: str, payload: dict) -> tuple[int, dict]:
        handler = object.__new__(ApiHandler)
        handler.server = SimpleNamespace(app=app)
        handler.path = path
        handler.headers = {
            "Content-Type": "application/json",
            "X-G1-Atom-Token": app.csrf_token,
        }
        handler._request_host_ok = lambda: True
        handler._body = lambda: payload
        responses: list[tuple[int, dict]] = []
        handler._json = lambda status, body: responses.append((status, body))
        handler.do_POST()
        self.assertEqual(len(responses), 1)
        return responses[0]

    def test_loads_content_addressed_catalog_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            loaded = load_catalog(self.write_catalog(Path(raw), catalog_payload()))
            self.assertEqual(len(loaded.entries), 1)
            atom = loaded.entries[0]
            self.assertAlmostEqual(atom.max_segment_velocity, 1.0)
            self.assertEqual(atom.motion_sha256, canonical_sha256(atom.motion))
            self.assertEqual(atom.bridge_from_neutral, 0.08)
            self.assertEqual(atom.bridge_to_neutral, 0.2)

    def test_rejects_non_content_addressed_filename(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            source = self.write_catalog(directory, catalog_payload())
            target = directory / "catalog.json"
            source.rename(target)
            with self.assertRaisesRegex(CatalogError, "content address"):
                load_catalog(target)

    def test_review_is_bound_to_exact_motion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            loaded = load_catalog(self.write_catalog(directory, catalog_payload()))
            store = ReviewStore(
                directory / "reviews.json", loaded, {"fixture.atom": "c" * 64}
            )
            record = store.set(loaded.entries[0], status="approved", reviewer="tester")
            self.assertTrue(record["current"])
            self.assertEqual(store.get(loaded.entries[0])["status"], "approved")
            reloaded = ReviewStore(
                directory / "reviews.json", loaded, {"fixture.atom": "d" * 64}
            )
            self.assertEqual(reloaded.get(loaded.entries[0])["status"], "stale")

    def test_validation_does_not_change_motion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            loaded = load_catalog(self.write_catalog(Path(raw), catalog_payload()))
            atom = loaded.entries[0]
            before = canonical_sha256(atom.motion)
            limits = {
                "source": "fixture",
                "position_margin_rad": 0.0,
                "joints": {
                    joint: {"min_rad": -2.0, "max_rad": 2.0}
                    for joint in EXPECTED_JOINTS
                },
            }
            self.assertTrue(validate_atom(atom, limits)["ok"])
            self.assertEqual(canonical_sha256(atom.motion), before)

    def test_validation_measures_but_does_not_gate_atom_segment_velocity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            payload = catalog_payload()
            payload["entries"][0]["motion"] = motion(last=0.01, delta=1.0)
            payload["entries"][0]["duration_us"] = 10000
            loaded = load_catalog(self.write_catalog(Path(raw), payload))
            atom = loaded.entries[0]
            limits = {
                "source": "fixture",
                "position_margin_rad": 0.0,
                "joints": {
                    joint: {"min_rad": -2.0, "max_rad": 2.0}
                    for joint in EXPECTED_JOINTS
                },
            }
            result = validate_atom(atom, limits)
            self.assertTrue(result["ok"])
            self.assertAlmostEqual(result["max_segment_velocity"], 100.0)
            self.assertFalse(result["atom_segment_velocity_gate_enabled"])

    def test_validation_still_blocks_joint_position_limit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            payload = catalog_payload()
            payload["entries"][0]["motion"] = motion(last=0.01, delta=1.5)
            payload["entries"][0]["duration_us"] = 10000
            loaded = load_catalog(self.write_catalog(Path(raw), payload))
            atom = loaded.entries[0]
            limits = {
                "source": "fixture",
                "position_margin_rad": 0.1,
                "joints": {
                    joint: {"min_rad": -1.5, "max_rad": 1.5}
                    for joint in EXPECTED_JOINTS
                },
            }
            result = validate_atom(atom, limits)
            self.assertFalse(result["ok"])
            self.assertEqual(result["errors"][0]["code"], "joint_position_limit")

    def test_limits_reject_legacy_atom_segment_velocity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "limits.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "g1_atom_limits_v1",
                        "position_margin_rad": 0.0,
                        "max_atom_segment_velocity_rad_s": 1.2,
                        "joints": {
                            joint: {"min_rad": -2.0, "max_rad": 2.0}
                            for joint in EXPECTED_JOINTS
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LimitsError, "removed legacy Atom-body gate"):
                load_limits(path)

    def test_minimum_jerk_duration_matches_sim_contract(self) -> None:
        self.assertEqual(minimum_jerk_duration(0.0), 0.08)
        self.assertEqual(minimum_jerk_duration(0.1), 0.2)
        self.assertEqual(minimum_jerk_duration(0.64), 1.2)

    def test_limits_reject_nan_margin(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "limits.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "g1_atom_limits_v1",
                        "position_margin_rad": float("nan"),
                        "joints": {
                            joint: {"min_rad": -2.0, "max_rad": 2.0}
                            for joint in EXPECTED_JOINTS
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LimitsError, "finite"):
                load_limits(path)

    def test_runtime_preview_replacement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "preview.mp4"
            original = b"reviewed-preview-bytes"
            path.write_bytes(original)
            app = object.__new__(AtomPlayerApp)
            app.previews = {
                "fixture.atom": PreviewAsset(
                    path=path, sha256=hashlib.sha256(original).hexdigest()
                )
            }
            loaded = load_catalog(self.write_catalog(Path(raw), catalog_payload()))
            atom = loaded.entries[0]
            self.assertTrue(app.preview_integrity_ok(atom))
            path.write_bytes(b"replacement")
            self.assertFalse(app.preview_integrity_ok(atom))
            with self.assertRaisesRegex(RuntimeError, "changed after startup"):
                app.require_preview_integrity(atom)

    def test_atom_payload_exposes_ordered_endpoints_without_mutating_motion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app = self.make_sequence_app(Path(raw))
            atom = app.catalog.entries[1]
            before = canonical_sha256(atom.motion)
            payload = app.atom_payload(atom)
            self.assertEqual(payload["first_q"], [0.0] * len(EXPECTED_JOINTS))
            self.assertEqual(payload["last_q"], [0.2] * len(EXPECTED_JOINTS))
            self.assertEqual(canonical_sha256(atom.motion), before)

    def test_prepare_sequence_requires_two_or_three_unique_exact_entry_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app = self.make_sequence_app(Path(raw))
            entry_ids = [atom.entry_id for atom in app.catalog.entries]
            self.assertEqual(
                [atom.entry_id for atom in app.prepare_sequence(entry_ids[:2])],
                entry_ids[:2],
            )
            self.assertEqual(
                [atom.entry_id for atom in app.prepare_sequence(entry_ids)],
                entry_ids,
            )
            with self.assertRaisesRegex(ValueError, "exactly 2 or 3"):
                app.prepare_sequence(entry_ids[:1])
            with self.assertRaisesRegex(ValueError, "exactly 2 or 3"):
                app.prepare_sequence(entry_ids + ["d" * 64])
            with self.assertRaisesRegex(ValueError, "duplicates"):
                app.prepare_sequence([entry_ids[0], entry_ids[0]])
            with self.assertRaisesRegex(ValueError, "lowercase 64-character"):
                app.prepare_sequence([entry_ids[0], "not-an-entry-id"])
            with self.assertRaisesRegex(LookupError, "sequence position 2"):
                app.prepare_sequence([entry_ids[0], "f" * 64])

    def test_prepare_sequence_rechecks_every_approval_preview_and_position_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app = self.make_sequence_app(Path(raw))
            first, second, third = app.catalog.entries

            app.reviews.set(second, status="rejected", reviewer="tester")
            with self.assertRaisesRegex(RuntimeError, "position 2.*not a current approved"):
                app.prepare_sequence([first.entry_id, second.entry_id])
            app.reviews.set(second, status="approved", reviewer="tester")

            app.previews[second.atom_id].path.write_bytes(b"changed after review")
            with self.assertRaisesRegex(RuntimeError, "position 2.*preview integrity"):
                app.prepare_sequence([first.entry_id, second.entry_id])
            preview = app.previews[second.atom_id]
            preview.path.write_bytes(f"reviewed preview for {second.atom_id}".encode())

            app.limits = {
                **app.limits,
                "joints": {
                    joint: {"min_rad": -2.0, "max_rad": 0.25}
                    for joint in EXPECTED_JOINTS
                },
            }
            with self.assertRaisesRegex(RuntimeError, "position 2.*hardware gate"):
                app.prepare_sequence([first.entry_id, third.entry_id])

    def test_sequence_api_preserves_order_and_rejects_invalid_or_unready_requests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app = self.make_sequence_app(Path(raw))
            entry_ids = [atom.entry_id for atom in app.catalog.entries]

            status, body = self.post_json(
                app, "/api/run/sequence", {"entry_ids": [entry_ids[2], entry_ids[0]]}
            )
            self.assertEqual(status, 202)
            self.assertTrue(body["ok"])
            self.assertEqual(
                [atom.entry_id for atom in app.runs.sequences[0]],
                [entry_ids[2], entry_ids[0]],
            )

            status, _ = self.post_json(
                app, "/api/run/sequence", {"entry_ids": [entry_ids[0], entry_ids[0]]}
            )
            self.assertEqual(status, 400)
            self.assertEqual(len(app.runs.sequences), 1)

            app.config.runner_path.chmod(0o644)
            status, _ = self.post_json(
                app, "/api/run/sequence", {"entry_ids": entry_ids[:2]}
            )
            self.assertEqual(status, 503)
            self.assertEqual(len(app.runs.sequences), 1)

    def test_run_manager_starts_one_process_with_ordered_exact_sequence_flags(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            catalog = load_catalog(self.write_catalog(directory, sequence_catalog_payload()))
            config = SimpleNamespace(
                runner_path=directory / "runner",
                catalog_path=catalog.path,
                limits_path=directory / "limits.json",
                network_interface="eno2",
            )
            audit_path = directory / "audit.jsonl"
            manager = RunManager(config, AuditLog(audit_path))
            process = mock.Mock(pid=4242)
            process.poll.return_value = None
            with mock.patch(
                "g1_atom_player.server.subprocess.Popen", return_value=process
            ) as popen, mock.patch("g1_atom_player.server.threading.Thread") as worker:
                snapshot = manager.start_sequence(catalog.entries)

            popen.assert_called_once()
            worker.return_value.start.assert_called_once()
            command = popen.call_args.args[0]
            self.assertEqual(
                [item.removeprefix("--entry-id=") for item in command if item.startswith("--entry-id=")],
                [atom.entry_id for atom in catalog.entries],
            )
            self.assertEqual(
                [
                    item.removeprefix("--expect-version=")
                    for item in command
                    if item.startswith("--expect-version=")
                ],
                [atom.version for atom in catalog.entries],
            )
            self.assertEqual(
                [
                    item.removeprefix("--expect-revision=")
                    for item in command
                    if item.startswith("--expect-revision=")
                ],
                [str(atom.revision) for atom in catalog.entries],
            )
            self.assertEqual(
                [
                    item.removeprefix("--expect-content-sha256=")
                    for item in command
                    if item.startswith("--expect-content-sha256=")
                ],
                [atom.content_sha256 for atom in catalog.entries],
            )
            self.assertEqual(
                [
                    item.removeprefix("--confirm-entry-id=")
                    for item in command
                    if item.startswith("--confirm-entry-id=")
                ],
                [atom.entry_id for atom in catalog.entries],
            )
            self.assertEqual(snapshot["mode"], "sequence")
            self.assertEqual(snapshot["sequence_length"], 3)
            self.assertEqual(snapshot["pid"], 4242)
            audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
            self.assertEqual(
                [record["event"] for record in audit],
                ["real_sequence_start_requested", "real_sequence_started"],
            )
            self.assertEqual(
                [entry["entry_id"] for entry in audit[0]["entries"]],
                [atom.entry_id for atom in catalog.entries],
            )

    def test_shutdown_gate_rejects_new_run_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manager = RunManager(None, AuditLog(Path(raw) / "audit.jsonl"))  # type: ignore[arg-type]
            manager.begin_shutdown()
            loaded = load_catalog(self.write_catalog(Path(raw), catalog_payload()))
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                manager.start(loaded.entries[0])


if __name__ == "__main__":
    unittest.main()
