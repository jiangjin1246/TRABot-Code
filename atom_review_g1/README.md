# ② Real-robot atom review (G1 Atom Player)

Review and play catalog atoms one at a time on the real Unitree G1, and record
approve/reject decisions. Usage: see the main
[README](../README.md#-usage).

```text
g1_atom_player/   web server, catalog loading/validation, review store
web/              review page
g1_bridge/        native runner (C++, unitree_sdk2)
config/           g1_limits.json (joint limits used by the runner)
scripts/          build_native.sh, start_web.sh, stop_web.sh, validate_catalog.py
tests/
```

Execution of one run: fresh measured joint state, legs and waist held, a
minimum-jerk bridge into the exact atom body (original joint values and
timestamps, piecewise linear), a bridge to `catalog/safe_neutral.json`, then the
arm SDK weight is released.
