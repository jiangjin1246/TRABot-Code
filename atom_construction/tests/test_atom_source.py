from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from atom_pipeline.source import (
    discover_sources,
    load_source_manifest,
    parse_source_identity,
    read_source_manifest_metadata,
    select_sources,
    write_source_manifest,
)


class AtomSourceTests(unittest.TestCase):
    def test_filename_identity_is_explicit(self) -> None:
        identity = parse_source_identity("21_ayana_1_12_12.pkl")
        self.assertEqual(identity.speaker_key, "21_ayana")
        self.assertEqual(identity.session_id, "1")
        self.assertEqual(identity.segment_id, "12")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            parse_source_identity("21_ayana_1_12_13.pkl")

    def test_selection_is_multispeaker_and_session_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for speaker in range(1, 7):
                for session in (0, 1):
                    for segment in (1, 2):
                        (root / f"{speaker}_person{speaker}_{session}_{segment}_{segment}.pkl").touch()
            records = discover_sources(root)
            selected = select_sources(
                records,
                max_speakers=4,
                sources_per_speaker=2,
                seed=20260915,
            )
        speakers = {record.identity.speaker_key for record in selected}
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(speakers), 4)
        for speaker in speakers:
            self.assertEqual(
                {record.identity.session_id for record in selected if record.identity.speaker_key == speaker},
                {"0", "1"},
            )

    def test_manifest_round_trip_requires_explicit_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pkl-results"
            root.mkdir()
            source = root / "1_wayne_0_1_1.pkl"
            source.touch()
            records = discover_sources(root)
            manifest = Path(directory) / "selected.json"
            write_source_manifest(manifest, records, selection={"seed": 7})
            loaded = load_source_manifest(manifest, source_root=root)
            metadata = read_source_manifest_metadata(manifest)
        self.assertEqual(loaded[0].relative_path, "1_wayne_0_1_1.pkl")
        self.assertEqual(loaded[0].identity, records[0].identity)
        self.assertEqual(metadata["original_selection"], {"seed": 7})
        self.assertEqual(len(metadata["replayed_source_manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
