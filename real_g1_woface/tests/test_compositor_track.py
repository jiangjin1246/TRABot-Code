from __future__ import annotations

import asyncio
from dataclasses import replace
from fractions import Fraction
import os
import unittest
from unittest import mock

import numpy as np

from web_overlay.compositor import CompositeVideoTrack, MediaStreamError
from web_overlay.config import OverlayConfig
from web_overlay.sim_client import CachedFrame


class _InputFrame:
    def __init__(self, image, *, pts=900, time_base=Fraction(1, 90_000)):
        self.image = image
        self.pts = pts
        self.time_base = time_base

    def to_ndarray(self, *, format):
        assert format == "bgr24"
        return self.image


class _OutputFrame:
    def __init__(self, image):
        self.image = image.copy()
        self.pts = None
        self.time_base = None


class _StallingSource:
    def __init__(self, frame):
        self.frame = frame
        self.calls = 0
        self.stops = 0
        self.release = asyncio.Event()

    async def recv(self):
        self.calls += 1
        if self.calls == 1:
            return self.frame
        await self.release.wait()
        raise MediaStreamError

    def stop(self):
        self.stops += 1


class _EndingSource(_StallingSource):
    pass


class _Client:
    def __init__(self):
        self.cached = None
        self.poll_calls = 0

    def ensure_frame_polling(self):
        self.poll_calls += 1

    def latest_frame(self):
        return self.cached


class _ManualClock:
    def __init__(self, value=10.0):
        self.value = value
        self.delays = []

    def __call__(self):
        return self.value

    async def sleep(self, delay):
        self.delays.append(delay)
        self.value += delay
        await asyncio.sleep(0)


class CompositeVideoTrackTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = replace(
            OverlayConfig(), panel_width=1, panel_height=1, composite_fps=25.0
        )

    async def test_stalled_face_repeats_at_fixed_rate_with_latest_sim(self):
        face = np.full((1, 1, 3), 11, dtype=np.uint8)
        sim_one = np.full((1, 1, 3), 21, dtype=np.uint8)
        sim_two = np.full((1, 1, 3), 31, dtype=np.uint8)
        source = _StallingSource(_InputFrame(face))
        client = _Client()
        clock = _ManualClock()
        client.cached = CachedFrame(b"one", "image/png", clock())
        track = CompositeVideoTrack(
            source,
            client,
            self.config,
            clock=clock,
            sleep=clock.sleep,
        )

        def decode(frame):
            return sim_one if frame.body == b"one" else sim_two

        try:
            with mock.patch(
                "web_overlay.compositor.decode_cached_frame", side_effect=decode
            ) as decode_mock, mock.patch(
                "web_overlay.compositor._video_frame_from_bgr",
                side_effect=_OutputFrame,
            ):
                first = await asyncio.wait_for(track.recv(), timeout=0.2)
                client.cached = CachedFrame(b"two", "image/png", clock())
                second = await asyncio.wait_for(track.recv(), timeout=0.2)

                self.assertEqual(source.calls, 2)
                np.testing.assert_array_equal(first.image[0, 0], [11, 11, 11])
                np.testing.assert_array_equal(first.image[0, 1], [21, 21, 21])
                np.testing.assert_array_equal(second.image[0, 0], [11, 11, 11])
                np.testing.assert_array_equal(second.image[0, 1], [31, 31, 31])
                self.assertEqual((first.pts, second.pts), (900, 4_500))
                self.assertEqual(first.time_base, Fraction(1, 90_000))
                self.assertEqual(second.time_base, Fraction(1, 90_000))
                self.assertEqual(client.poll_calls, 1)
                self.assertEqual(decode_mock.call_count, 2)
                self.assertAlmostEqual(clock.delays[0], 1 / 25, places=7)

                # A 200 ms scheduler delay skips expired ticks.  The following
                # recv sleeps for the next deadline instead of burst-catching up.
                clock.value += 0.2
                late = await track.recv()
                delay_count_after_late_frame = len(clock.delays)
                following = await track.recv()
                self.assertGreater(late.pts, second.pts + 3_600)
                self.assertEqual(len(clock.delays), delay_count_after_late_frame + 1)
                self.assertGreater(following.pts, late.pts)
        finally:
            track.stop()
            await asyncio.sleep(0)

        self.assertEqual(source.stops, 1)

    async def test_source_end_is_propagated(self):
        face = np.full((1, 1, 3), 7, dtype=np.uint8)
        source = _EndingSource(_InputFrame(face, pts=0))
        client = _Client()
        clock = _ManualClock()
        track = CompositeVideoTrack(
            source,
            client,
            self.config,
            clock=clock,
            sleep=clock.sleep,
        )

        try:
            with mock.patch(
                "web_overlay.compositor._video_frame_from_bgr",
                side_effect=_OutputFrame,
            ):
                await track.recv()
                source.release.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                with self.assertRaises(MediaStreamError):
                    await track.recv()
        finally:
            track.stop()
            await asyncio.sleep(0)

    async def test_stop_cancels_reader_and_ends_track(self):
        face = np.full((1, 1, 3), 5, dtype=np.uint8)
        source = _StallingSource(_InputFrame(face, pts=0))
        track = CompositeVideoTrack(source, _Client(), self.config)

        with mock.patch(
            "web_overlay.compositor._video_frame_from_bgr",
            side_effect=_OutputFrame,
        ):
            await track.recv()
            track.stop()
            with self.assertRaises(MediaStreamError):
                await track.recv()
        await asyncio.sleep(0)
        self.assertEqual(source.stops, 1)


class CompositeConfigTest(unittest.TestCase):
    def test_composite_fps_environment_override(self):
        with mock.patch.dict(os.environ, {"OVERLAY_COMPOSITE_FPS": "30"}):
            self.assertEqual(OverlayConfig.from_env().composite_fps, 30.0)

    def test_non_positive_composite_fps_is_rejected(self):
        config = replace(OverlayConfig(), composite_fps=0)
        with self.assertRaisesRegex(ValueError, "composite_fps"):
            CompositeVideoTrack(object(), _Client(), config)


if __name__ == "__main__":
    unittest.main()
