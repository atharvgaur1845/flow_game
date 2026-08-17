"""Emit a golden trace of (o_t, velocity) -> pi from the torch backend.

Used by tools/verify_port.mjs to prove the JavaScript port in docs/js/
reproduces AttentionStateInference exactly. Deterministic input, no RNG.
"""
import json
import math
import os
import sys

os.environ["FLOW_HEADLESS"] = "1"
sys.path.insert(0, "/home/atharv/Desktop/projects/flow_game")

import torch  # noqa: E402

from flow_game import AttentionStateInference  # noqa: E402

attn = AttentionStateInference()
trace = []

for f in range(120):
    # A deterministic, deliberately varied observation stream: fast motion,
    # idling, threat spikes — enough to exercise every branch of
    # project_future (including the idle-growth path).
    vx = math.sin(f * 0.17) * 3.0 * (0.0 if 40 <= f < 60 else 1.0)
    vy = math.cos(f * 0.11) * 2.0 * (0.0 if 40 <= f < 60 else 1.0)
    v_norm = math.hypot(vx, vy)
    apm = 4.0 * abs(math.sin(f * 0.05))
    dir_var = 0.5 + 0.5 * math.sin(f * 0.23)
    threat = 1.0 / (1.0 + abs(math.sin(f * 0.07)) * 12.0)
    idle = min(max((f - 40) / 20.0, 0.0), 1.0) if f < 60 else 0.0

    o_t = torch.tensor([v_norm, apm, dir_var, threat, idle],
                       dtype=torch.float32)
    pi = attn(o_t, (vx, vy))
    trace.append({
        "o": [float(x) for x in o_t.tolist()],
        "v": [vx, vy],
        "pi": [float(x) for x in pi.tolist()],
    })

print(json.dumps(trace))
