"""Dump the backend inference weights to a JS module so the web port is
numerically identical to the Python game."""
import os
import sys

os.environ["FLOW_HEADLESS"] = "1"
sys.path.insert(0, "/home/atharv/Desktop/projects/flow_game")

from flow_game import AttentionStateInference, EnvironmentMapper  # noqa: E402

attn = AttentionStateInference()
mapper = EnvironmentMapper()


def fmt(t, name):
    a = t.detach().cpu().numpy()
    if a.ndim == 1:
        body = "[" + ", ".join(f"{v:.8g}" for v in a) + "]"
    else:
        rows = ["  [" + ", ".join(f"{v:.8g}" for v in row) + "]" for row in a]
        body = "[\n" + ",\n".join(rows) + "\n]"
    return f"export const {name} = {body};\n"


out = [
    "// AUTO-GENERATED from flow_game.py — do not edit by hand.\n",
    "// Exact weights of AttentionStateInference (torch.manual_seed(1337))\n",
    "// and EnvironmentMapper, so the web port infers the same pi as the\n",
    "// desktop build. Regenerate with tools/dump_weights.py.\n\n",
    fmt(attn.W_Q, "W_Q"),
    fmt(attn.W_K, "W_K"),
    fmt(attn.W_V, "W_V"),
    fmt(attn.W_pi, "W_PI"),
    fmt(attn.b_pi, "B_PI"),
    fmt(mapper.M, "M"),
    fmt(mapper.W_c1, "W_C1"),
    fmt(mapper.W_c2, "W_C2"),
]
sys.stdout.write("".join(out))
