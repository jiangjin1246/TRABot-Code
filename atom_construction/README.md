# ① Offline atom construction

Extraction, constraint validation and repair of motion atoms (paper §III-B).
Usage: see the main [README](../README.md#-usage).

```text
atom_pipeline/     source selection, activity segmentation, extraction, safety
                   checks, repair / auto-repair, Atom Schema v1 (schema/)
scripts/           extract_beat2_atoms.sh
tests/             unit tests (python -m unittest discover -s tests -t .)
```
