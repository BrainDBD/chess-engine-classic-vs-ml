import argparse
import json
from pathlib import Path
import torch


def _cfloat(v) -> str:
    # %.9g can yield integer-looking tokens like "0" or "1", which become the invalid C++ literal "0f"/"1f". Ensure a decimal point so the 'f' suffix is always a valid float literal
    s = f"{v:.9g}"
    if not any(c in s for c in (".", "e", "E", "n", "i")):  # n/i guard nan/inf
        s += ".0"
    return s + "f"


def fmt_floats(vals, per_line=6, indent="    "):
    out, line = [], []
    for _, v in enumerate(vals):
        line.append(_cfloat(v))
        if len(line) == per_line:
            out.append(indent + ", ".join(line) + ",")
            line = []
    if line:
        out.append(indent + ", ".join(line) + ",")
    return "\n".join(out)


def emit_matrix(name, mat, cols):
    # mat: list of rows (each `cols` long) - C++ float[rows][cols]
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


def _softplus(x: float) -> float:
    # numerically stable softplus, matching torch.nn.functional.softplus
    import math
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def resolved_ordered_biases(sd):
    """Reconstruct the C++-facing [b0, b1] from the reparameterized params:
    b1 = bias_low, b0 = b1 + softplus(bias_gap). Mirrors
    EndgameOrdinalNet.ordered_biases() so the header matches the trained model."""
    b1 = float(sd["bias_low"].item() if hasattr(sd["bias_low"], "item")
               else float(sd["bias_low"]))
    gap = float(sd["bias_gap"].item() if hasattr(sd["bias_gap"], "item")
                else float(sd["bias_gap"]))
    b0 = b1 + _softplus(gap)
    return [b0, b1]


def common_header(IN, H1, H2):
    return f"""
#pragma once

static const int EG_NET_IN = {IN};
static const int EG_NET_H1 = {H1};
static const int EG_NET_H2 = {H2};
"""

def build_ordinal(sd, names):
    mean = sd["feat_mean"].tolist()
    std = sd["feat_std"].tolist()
    W1 = sd["trunk.0.weight"].tolist(); b1 = sd["trunk.0.bias"].tolist()
    W2 = sd["trunk.3.weight"].tolist(); b2 = sd["trunk.3.bias"].tolist()
    # shared layer: Linear(H2, 1, bias=False) -> weight shape (1, H2); no bias.
    W_shared = sd["shared.weight"].tolist()[0]
    thresh_bias = resolved_ordered_biases(sd)   # resolved [b0, b1], b0 > b1 guaranteed

    IN, H1, H2 = len(mean), len(b1), len(b2)
    assert len(std) == IN and len(W1) == H1 and all(len(r) == IN for r in W1)
    assert len(W2) == H2 and all(len(r) == H1 for r in W2)
    assert len(W_shared) == H2, f"shared layer wants {H2}, got {len(W_shared)}"
    assert len(thresh_bias) == 2, (
        f"expected 2 ordinal thresholds (WDL), got {len(thresh_bias)}")
    assert len(names) == IN, f"norm.json has {len(names)} names, net wants {IN}"
    # Ordinality is now structural (softplus reparameterization), so b0 > b1 must
    # hold. Assert it rather than warn -- a failure means a parameterization bug.
    assert thresh_bias[0] > thresh_bias[1], (
        f"ordered biases not descending: {thresh_bias} -- parameterization bug")

    parts = [common_header(IN, H1, H2)]
    parts.append(f"static const int EG_NET_THRESHOLDS = {len(thresh_bias)};")
    parts.append(emit_vector("EG_FEAT_MEAN", mean))
    parts.append(emit_vector("EG_FEAT_STD", std))
    parts.append(emit_matrix("EG_W1", W1, IN))
    parts.append(emit_vector("EG_B1", b1))
    parts.append(emit_matrix("EG_W2", W2, H1))
    parts.append(emit_vector("EG_B2", b2))
    # names match endgame_net_ordinal.h: EG_W_SHARED + EG_THRESH_BIAS
    parts.append(emit_vector("EG_W_SHARED", W_shared))
    parts.append(emit_vector("EG_THRESH_BIAS", thresh_bias))
    namelist = ",\n    ".join(f'"{n}"' for n in names)
    parts.append(f"static const char* EG_FEATURE_ORDER[{IN}] = {{\n    {namelist}\n}};")
    summary = (f"{IN}->{H1}->{H2}->1 shared logit + {len(thresh_bias)} ordinal "
               f"thresholds, {IN*H1 + H1*H2 + H2} weights + {H1+H2} biases "
               f"+ {len(thresh_bias)} thresholds")
    return parts, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("--norm", required=True, help="the .norm.json (feature order)")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "src" / "net" / "endgame_net_weights.h"),
    )
    cfg = ap.parse_args()

    sd = torch.load(cfg.pt, map_location="cpu")
    norm = json.loads(Path(cfg.norm).read_text())
    names = norm["feature_order"]

    parts, summary = build_ordinal(sd, names)

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # create src/net/ if missing
    out_path.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {cfg.out}: {summary}")
    print("  -> include this from endgame_net.h "
            "(expects EG_W_SHARED + EG_THRESH_BIAS)")


if __name__ == "__main__":
    main()