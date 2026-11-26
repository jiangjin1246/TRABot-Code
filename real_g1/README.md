# ④ Real G1 with face

The `sim/` dialogue, planner and executor driving the real Unitree G1 through a
pinned native runner; the browser shows the FlashHead face. Only atoms approved
in `../catalog/g1_reviews.json` are played. Usage and safety notes: see the main
[README](../README.md#-usage).

```text
app.py, web_overlay/   as in sim/, with the g1_realtime backend (g1_session_client.py)
g1_bridge/             native runner g1_atom_motion_cli (C++, unitree_sdk2)
config/g1_limits.json  joint limits enforced by the runner
scripts/               build_g1_bridge.sh, start_realtime.sh, stop_realtime.sh
tests/
```

Per reply: one native control session; for each planned atom the runner takes
a fresh measured state, bridges with minimum jerk into the exact atom, replans
the remaining suffix, and ends with one canonical neutral and release.
