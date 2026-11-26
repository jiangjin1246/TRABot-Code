# ③ Simulation

TRABot with a MuJoCo G1: Gemini Live dialogue, semantic-conditioned planner,
state-aware executor, and the FlashHead face rendered onto the robot's head.
Usage: see the main [README](../README.md#-usage).

```text
app.py            launcher: loads ../face/webrtc_app_gemini_http.py and patches it
web_overlay/      planner, executor, dialogue session, compositor, experiment recording
sim_service/      persistent fixed-base MuJoCo service (motion queue, frames)
prompts/          Gemini system prompt for semantic gestures
scripts/          start_all.sh, stop_all.sh, run_sim.sh, run_web.sh, setup_links.sh
tests/
```

`scripts/setup_links.sh` (run by `start_all.sh`) links `../face` (code,
weights, reference images) and `../catalog` (neutral pose, legacy gestures)
into this folder.
