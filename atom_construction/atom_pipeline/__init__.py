"""BEAT2-derived gesture atom authoring tools.

The package is deliberately offline and simulator-first.  It turns trusted,
already-retargeted G1 trajectories into reviewable Atom Schema v1 candidates;
it is not a real-robot motion controller.
"""

from .segmentation import DetectorConfig, GestureSegment, detect_gesture_segments
from .source import SourceIdentity, SourceRecord, discover_sources, select_sources

__all__ = [
    "DetectorConfig",
    "GestureSegment",
    "SourceIdentity",
    "SourceRecord",
    "detect_gesture_segments",
    "discover_sources",
    "select_sources",
]
