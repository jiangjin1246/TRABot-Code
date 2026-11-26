# ⑤ Real G1 without face (CPU only)

Same as `../real_g1/` but `web_overlay/faceless.py` replaces FlashHead
inference before the face app is imported: every 0.96 s audio slice still flows
through the original chunk/playback loop (same reply timing, barge-in, trace
and recording), while the video track carries a blank frame and no face weights
are loaded. `run_web.sh` sets `CUDA_VISIBLE_DEVICES=""`.

Install the Python packages from `requirements-woface.txt`, then follow step ④
of the main [README](../README.md#-usage).
`real_g1/` and `real_g1_woface/` use the same ports (1131/8792); run one at a time.
