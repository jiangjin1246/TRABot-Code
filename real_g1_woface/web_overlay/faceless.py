"""CPU-only stand-in for ``flash_head.inference``: no face is generated.

The face app (../face) paces Gemini audio through the FlashHead chunk loop:
every slice of reply speech becomes one video chunk that is published together
with its audio. Reply timing, barge-in, the reply-trace hub and the experiment
recorder all hang off that loop, so it is kept unchanged. Only the diffusion
model is replaced: each chunk returns one constant blank frame per video slot,
the process never loads FlashHead/wav2vec weights and never initialises CUDA.
"""

from __future__ import annotations

import copy
import sys
from types import ModuleType
from typing import Any

import torch

# Same values as face/flash_head/configs/
# infer_params.yaml. motion_frames_num is what get_pipeline() derives for the
# Model_Pro checkpoint: 33 - 9 = 24 frames (15360 samples, 0.96 s) per slice,
# matching the chunks logged by the face-enabled deployment.
_INFER_PARAMS: dict[str, Any] = {
    "frame_num": 33,
    "motion_frames_latent_num": 2,
    "motion_frames_num": 9,
    "tgt_fps": 25,
    "sample_rate": 16000,
    "sample_shift": 5,
    "color_correction_strength": 1.0,
    "cached_audio_duration": 8,
    "num_heads": 12,
    "height": 512,
    "width": 512,
    "sample_steps": 4,
}

# Dark grey keeps the existing video element visibly "on" without a face.
BLANK_FRAME_VALUE = 24


class FacelessPipeline:
    """Holds the one cached blank chunk returned for every audio slice."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.frames = torch.full(
            (
                _INFER_PARAMS["frame_num"],
                _INFER_PARAMS["height"],
                _INFER_PARAMS["width"],
                3,
            ),
            BLANK_FRAME_VALUE,
            dtype=torch.uint8,
        )


def get_pipeline(world_size: int, ckpt_dir: str, model_type: str, wav2vec_dir: str):
    return FacelessPipeline()


def get_base_data(pipeline: Any, cond_image_path_or_dir: Any, base_seed: Any, use_face_crop: Any):
    return None


def get_infer_params() -> dict[str, Any]:
    return copy.deepcopy(_INFER_PARAMS)


def get_audio_embedding(pipeline: Any, audio_array: Any, audio_start_idx: int = -1, audio_end_idx: int = -1):
    return None


def run_pipeline(pipeline: FacelessPipeline, audio_embedding: Any):
    return pipeline.frames


def install_faceless_inference_stub() -> ModuleType:
    """Register this module as ``flash_head.inference`` before upstream import."""

    package = ModuleType("flash_head")
    package.__path__ = []  # a package that can never resolve the real submodules
    package.__file__ = "<flashhead-faceless package>"
    inference = ModuleType("flash_head.inference")
    for name in (
        "get_pipeline",
        "get_base_data",
        "get_infer_params",
        "get_audio_embedding",
        "run_pipeline",
    ):
        setattr(inference, name, globals()[name])
    inference.__file__ = "<flashhead-faceless inference>"
    package.inference = inference
    sys.modules["flash_head"] = package
    sys.modules["flash_head.inference"] = inference
    return inference
