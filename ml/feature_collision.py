import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import chess

sys.path.insert(0, str(Path(__file__).parent))
# reuse training's exact split + model machinery so the validation rows match.
from train_endgame import (EndgameDataset, group_split, label_to_class,
                           EndgameOrdinalNet, ordinal_accuracy,
                           run_detailed_report, DEVICE)
from endgame_features import endgame_features


def ceiling_on(X, cls, idx):
    # Occurrence-weighted feature ceiling over the given row indices
    buckets = defaultdict(Counter)
    for i in idx:
        buckets[X[i].tobytes()][int(cls[i])] += 1
    n = len(idx)
    correct = sum(cc.most_common(1)[0][1] for cc in buckets.values())
    return (correct / n if n else float("nan")), len(buckets)


def verify_integer_features(X):
    frac = np.abs(X - np.rint(X))
    max_frac = float(frac.max()) if X.size else 0.0
    return max_frac == 0.0, max_frac


def verify_source_parity(jsonl, X, k=2000, max_samples=0):
    checked = bad = 0
    with open(jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if (max_samples and i >= max_samples) or i >= k:
                break
            d = endgame_features(chess.Board(json.loads(line)["fen"]))
            if any(not isinstance(v, int) for v in d.values()):
                bad += 1
            elif tuple(d.values()) != tuple(int(round(x)) for x in X[i]):
                bad += 1
            checked += 1
    return checked, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="MUST match training (default 0.1)")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated split seeds; the FIRST must match the "
                         "seed used in training (default 42). Extra seeds are used "
                         "only for ceiling-stability checks")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="MUST match training (0 = no cap)")
    ap.add_argument("--load", default="",
                    help="trained .pt checkpoint (optional); compared on the FIRST seed only")
    cfg = ap.parse_args(argv)
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]

    print(f"Loading {cfg.jsonl} ...")
    ds = EndgameDataset(cfg.jsonl, cfg.max_samples)
    if (ds.game_ids == -1).all():
        sys.exit("[fatal] no usable game_id -> cannot reproduce a leak-free split.")
    X = ds.eg_feats
    true_cls = label_to_class(ds.labels)
    print(f"  {len(ds.labels):,} positions, {X.shape[1]} features")

    # --- correctness checks (so the ceiling number is defensible) ----------------
    clean, max_frac = verify_integer_features(X)
    checked, bad = verify_source_parity(cfg.jsonl, X, max_samples=cfg.max_samples)
    print("\n== Correctness checks ==")
    print(f"  features integer-valued      : {clean}  (max fractional part {max_frac:g})")
    print(f"  source parity (first {checked:,})  : {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    if not clean:
        print("  [warn] non-integer feature found -> inspect for float jitter "
              "before quoting the floor.")
    if bad:
        print("  [warn] stored features disagree with endgame_features() -> "
              "investigate before quoting any number.")

    # --- primary seed: full collision report + model comparison ------------------
    primary = seeds[0]
    _, val_idx = group_split(ds.game_ids, cfg.val_frac, seed=primary)
    val_set = set(val_idx.tolist())

    fens_by_feat = defaultdict(dict)   # feature-bytes -> {fen: class}
    fen_conflict = nval = 0
    with open(cfg.jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if cfg.max_samples and i >= cfg.max_samples:
                break
            if i not in val_set:
                continue
            rec = json.loads(line)
            fen, c = rec["fen"], int(true_cls[i])
            d = fens_by_feat[X[i].tobytes()]
            if fen in d and d[fen] != c:
                fen_conflict += 1
            d[fen] = c
            nval += 1
    if nval != len(val_idx):
        sys.exit(f"[fatal] row mismatch: {nval} re-read vs {len(val_idx)} in split.")

    P = sum(len(d) for d in fens_by_feat.values())
    collisions = [d for d in fens_by_feat.values() if len(d) >= 2]
    conflicting = [d for d in collisions if len(set(d.values())) >= 2]
    pos_coll = sum(len(d) for d in collisions)
    pos_conf = sum(len(d) for d in conflicting)
    ceil_primary, U = ceiling_on(X, true_cls, val_idx)

    print(f"\n== Validation set, seed={primary} ({len(val_idx):,} rows) ==")
    print(f"  distinct positions           : {P:,}")
    print(f"  distinct feature vectors     : {U:,}")
    print(f"  positions in collisions      : {pos_coll:,} ({100 * pos_coll / P:.1f}%)")
    print(f"  label-conflicting            : {len(conflicting):,} vectors, "
          f"{pos_conf:,} positions")
    print(f"  feature ceiling              : {ceil_primary:.4f}")
    print(f"  irreducible floor            : {1 - ceil_primary:.4f}")
    if fen_conflict:
        print(f"  [warn] {fen_conflict} same-FEN label conflict(s) -> data noise, "
              "not feature aliasing.")

    if cfg.load:
        dim = X.shape[1]
        model = EndgameOrdinalNet(torch.zeros(dim), torch.ones(dim)).to(DEVICE)
        model.load_state_dict(torch.load(cfg.load, map_location=DEVICE))
        ach = ordinal_accuracy(model, X, true_cls, val_idx)
        print(f"\n== Model ({cfg.load}) on seed={primary} val ==")
        print(f"  achieved accuracy            : {ach:.4f}")
        print(f"  ceiling                      : {ceil_primary:.4f}")
        print(f"  recoverable gap              : {ceil_primary - ach:.4f}")
        print(f"  irreducible floor            : {1 - ceil_primary:.4f}")
        run_detailed_report(model, X, true_cls, val_idx, tag=f"val seed={primary}")

    # --- ceiling stability across seeds (representation check, not a model test) -
    if len(seeds) > 1:
        print("\n== Ceiling stability across split seeds ==")
        ceils = []
        for s in seeds:
            _, vi = group_split(ds.game_ids, cfg.val_frac, seed=s)
            c, _ = ceiling_on(X, true_cls, vi)
            ceils.append(c)
            print(f"  seed {s:<6} val={len(vi):>8,}  ceiling={c:.4f}")
        print(f"  -> range {min(ceils):.4f}-{max(ceils):.4f}  "
              f"(spread {max(ceils) - min(ceils):.4f})")
        print(f"  note: model accuracy is valid only for the training seed "
              f"({primary}); other seeds' val rows overlap training, so their "
              "ceilings test\n        only the stability of the representation, "
              "not the model.")


if __name__ == "__main__":
    main()