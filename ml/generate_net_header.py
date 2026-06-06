import argparse
import json
from pathlib import Path
import torch


def fmt_floats(vals, per_line=6, indent="    "):
    out, line = [], []
    for _, v in enumerate(vals):
        line.append(f"{v:.9g}f")
        if len(line) == per_line:
            out.append(indent + ", ".join(line) + ",")
            line = []
    if line:
        out.append(indent + ", ".join(line) + ",")
    return "\n".join(out)


def emit_matrix(name, mat, cols):
    # mat: list of rows (each `cols` long). C++ float[rows][cols].
    rows = len(mat)
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for r in mat:
        lines.append("  {")
        lines.append(fmt_floats(r))
        lines.append("  },")
    lines.append("};")
    return "\n".join(lines)


def emit_vector(name, vec):
    lines = [f"static const float {name}[{len(vec)}] = {{"]
    lines.append(fmt_floats(vec))
    lines.append("};")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("--norm", required=True, help="the .norm.json (feature order)")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "src" / "endgame_net_weights.h"),
    )
    cfg = ap.parse_args()

    sd = torch.load(cfg.pt, map_location="cpu")
    norm = json.loads(Path(cfg.norm).read_text())
    names = norm["feature_order"]

    mean = sd["feat_mean"].tolist()
    std = sd["feat_std"].tolist()
    W1 = sd["net.0.weight"].tolist(); b1 = sd["net.0.bias"].tolist()
    W2 = sd["net.3.weight"].tolist(); b2 = sd["net.3.bias"].tolist()
    W3 = sd["net.6.weight"].tolist()[0]; b3 = float(sd["net.6.bias"].tolist()[0])

    IN, H1, H2 = len(mean), len(b1), len(b2)
    # integrity checks -- fail loudly rather than emit a silently-wrong header
    assert len(std) == IN and len(W1) == H1 and all(len(r) == IN for r in W1)
    assert len(W2) == H2 and all(len(r) == H1 for r in W2)
    assert len(W3) == H2
    assert len(names) == IN, f"norm.json has {len(names)} names, net wants {IN}"

    parts = []
    parts.append(f"""
#pragma once

static const int EG_NET_IN = {IN};
static const int EG_NET_H1 = {H1};
static const int EG_NET_H2 = {H2};
""")
    parts.append(emit_vector("EG_FEAT_MEAN", mean))
    parts.append(emit_vector("EG_FEAT_STD", std))
    parts.append(emit_matrix("EG_W1", W1, IN))
    parts.append(emit_vector("EG_B1", b1))
    parts.append(emit_matrix("EG_W2", W2, H1))
    parts.append(emit_vector("EG_B2", b2))
    parts.append(emit_vector("EG_W3", W3))
    parts.append(f"static const float EG_B3 = {b3:.9g}f;")
    # feature names for the parity harness / documentation
    namelist = ",\n    ".join(f'"{n}"' for n in names)
    parts.append(f"static const char* EG_FEATURE_ORDER[{IN}] = {{\n    {namelist}\n}};")

    Path(cfg.out).write_text("\n\n".join(parts) + "\n")
    print(f"wrote {cfg.out}: {IN}->{H1}->{H2}->1, "
            f"{IN*H1 + H1*H2 + H2} weights + {H1+H2+1} biases")


if __name__ == "__main__":
    main()