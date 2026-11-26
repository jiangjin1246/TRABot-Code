import sys
import unittest

import numpy as np

from web_overlay import faceless


class FacelessInferenceTest(unittest.TestCase):
    def setUp(self):
        self._saved = {
            name: sys.modules.get(name) for name in ("flash_head", "flash_head.inference")
        }

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_stub_replaces_flash_head_import(self):
        faceless.install_faceless_inference_stub()
        from flash_head.inference import get_infer_params, get_pipeline, run_pipeline

        self.assertIs(run_pipeline, faceless.run_pipeline)
        with self.assertRaises(ImportError):
            import flash_head.src  # noqa: F401  real model code must stay unreachable
        params = get_infer_params()
        pipeline = get_pipeline(1, "unused", "pro", "unused")
        self.assertEqual(str(pipeline.device), "cpu")
        self.assertEqual(params["sample_rate"], 16000)
        self.assertEqual(params["tgt_fps"], 25)

    def test_chunk_matches_upstream_slicing(self):
        # Mirrors webrtc_app_gemini_http.StreamSession._run_live_generation.
        params = faceless.get_infer_params()
        pipeline = faceless.get_pipeline(1, "", "pro", "")
        video = faceless.run_pipeline(pipeline, faceless.get_audio_embedding(pipeline, None))
        video = video[params["motion_frames_num"]:]
        frames_np = video.cpu().numpy().astype(np.uint8)
        slice_len = params["frame_num"] - params["motion_frames_num"]
        speech_slice_len = slice_len * params["sample_rate"] // params["tgt_fps"]
        self.assertEqual(frames_np.shape, (24, 512, 512, 3))
        self.assertEqual(speech_slice_len, 15360)

    def test_infer_params_are_copies(self):
        params = faceless.get_infer_params()
        params["frame_num"] = 0
        self.assertEqual(faceless.get_infer_params()["frame_num"], 33)


if __name__ == "__main__":
    unittest.main()
