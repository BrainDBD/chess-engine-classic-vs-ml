import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_endgame import EndgameDataset, group_split, DEVICE
# reuse the ordinal model + label encoder from the training module
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_endgame", str(Path(__file__).parent / "train_endgame.py"))
_tord = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tord)
EndgameOrdinalNet = _tord.EndgameOrdinalNet
label_to_class = _tord.label_to_class


def verdict_from_probs(o0, o1, t0, t1):
    # Class {0,1,2} from the two threshold probabilities at thresholds (t0,t1)
    return (o0 > t0).astype(int) + (o1 > t1).astype(int)


def recalls(true, pred):
    out = []
    for c in range(3):
        m = true == c
        out.append((pred[m] == c).mean() if m.any() else float("nan"))
    return out  # [loss, draw, win]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("pt")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--grid", nargs=3, type=float, metavar=("LO", "HI", "STEP"),
                    default=[0.35, 0.65, 0.05],
                    help="threshold grid: from LO to HI in STEP increments")
    cfg = ap.parse_args(argv)

    ds = EndgameDataset(cfg.jsonl, cfg.max_samples)
    _, val_idx = group_split(ds.game_ids, cfg.val_frac)
    X = ds.eg_feats[val_idx]
    true = label_to_class(ds.labels[val_idx])

    dim = X.shape[1]
    model = EndgameOrdinalNet(torch.zeros(dim), torch.ones(dim)).to(DEVICE)
    model.load_state_dict(torch.load(cfg.pt, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    o0, o1 = probs[:, 0], probs[:, 1]

    counts = np.bincount(true, minlength=3)
    n = counts.sum()

    # --- baseline at (0.5, 0.5) ---
    base_pred = verdict_from_probs(o0, o1, 0.5, 0.5)
    base_acc = (base_pred == true).mean()
    base_rec = recalls(true, base_pred)
    print(f"\nValidation set: {n:,} positions  "
        f"(loss={counts[0]:,}  draw={counts[1]:,}  win={counts[2]:,})")
    print(f"\nBaseline (t0=t1=0.50):  acc={base_acc:.3f}  "
        f"loss={base_rec[0]:.3f}  draw={base_rec[1]:.3f}  win={base_rec[2]:.3f}")

    # --- sweep ---
    lo, hi, step = cfg.grid
    grid = np.round(np.arange(lo, hi + 1e-9, step), 4)
    print(f"\nSweep (t0 = '>=draw' thresh, t1 = '>=win' thresh):")
    print(f"  {'t0':>5} {'t1':>5}  {'acc':>6}  {'loss':>6} {'draw':>6} {'win':>6}  "
        f"{'min-recall':>10}  notes")

    rows = []
    for t0 in grid:
        for t1 in grid:
            pred = verdict_from_probs(o0, o1, t0, t1)
            acc = (pred == true).mean()
            r = recalls(true, pred)
            rows.append((t0, t1, acc, r))

    # Print a readable subset: the diagonal-ish region plus the best by two criteria.
    # 1) best overall accuracy
    best_acc = max(rows, key=lambda x: x[2])
    # 2) best balanced (maximize the MINIMUM per-class recall -> most even profile)
    best_bal = max(rows, key=lambda x: min(x[3]))

    for t0, t1, acc, r in rows:
        tag = ""
        if (t0, t1) == best_acc[:2]: tag += " <-best acc"
        if (t0, t1) == best_bal[:2]: tag += " <-best balanced"
        # only print the informative band to keep output readable
        if abs(t0 - 0.5) <= 0.15 and abs(t1 - 0.5) <= 0.15:
            print(f"  {t0:>5.2f} {t1:>5.2f}  {acc:>6.3f}  "
                f"{r[0]:>6.3f} {r[1]:>6.3f} {r[2]:>6.3f}  {min(r):>10.3f}{tag}")

    print("\n--- recommended operating points ---")
    for label, (t0, t1, acc, r) in (("best overall accuracy", best_acc),
                                    ("best balanced (max-min recall)", best_bal)):
        print(f"  {label}: t0={t0:.2f} t1={t1:.2f}  "
            f"acc={acc:.3f}  loss={r[0]:.3f} draw={r[1]:.3f} win={r[2]:.3f}")

    t0, t1 = best_bal[0], best_bal[1]
    print(f"\nIf you adopt the balanced point (t0={t0:.2f}, t1={t1:.2f}), update C++ "
        f"wdlVerdict() in endgame_net_ordinal.h to:")
    print(f"    int passed = (o[0] > {t0:.2f}f ? 1 : 0) + (o[1] > {t1:.2f}f ? 1 : 0);")
    print("  (and keep the Python verdict at the same thresholds for parity).")
    print("\nNote: the model is unchanged; this only moves the decision boundary. "
        "Report whichever operating point you deploy.")


if __name__ == "__main__":
    main()