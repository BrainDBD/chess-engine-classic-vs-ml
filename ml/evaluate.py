import argparse
import collections
import numpy as np
import train_endgame as T
from plot_confusion_matrix_wdl import plot_confusion_matrix


# signature is canonical (Q,R,B,N,P) and color-independent (stronger side first).
_PIECES = [("queens", "Q", 9), ("rooks", "R", 5), ("bishops", "B", 3),
           ("knights", "N", 3), ("pawns", "P", 1)]


def _counts(ds):
    nm = {n: i for i, n in enumerate(ds.eg_names)}
    X = ds.eg_feats
    n = len(X)
    wv = np.zeros(n, dtype=np.float32)
    bv = np.zeros(n, dtype=np.float32)
    wc, bc = {}, {}
    for stem, _, val in _PIECES:
        w = X[:, nm[stem + "_stm"]]
        b = X[:, nm[stem + "_opp"]]
        wc[stem], bc[stem] = w, b
        wv += w * val
        bv += b * val
    return wv, bv, wc, bc


def material_value_diff(ds):
    wv, bv, _, _ = _counts(ds)
    return np.abs(wv - bv), wv, bv


def signatures(ds) -> np.ndarray:
    wv, bv, wc, bc = _counts(ds)
    n = len(wv)
    # integer count matrix, columns in _PIECES order, white then black
    cols = ([np.rint(wc[stem]).astype(int) for stem, _, _ in _PIECES]
            + [np.rint(bc[stem]).astype(int) for stem, _, _ in _PIECES])
    mat = np.stack(cols, axis=1)
    uniq, inv = np.unique(mat, axis=0, return_inverse=True)
    letters = [ch for _, ch, _ in _PIECES]
    vals = np.array([v for _, _, v in _PIECES])
    sig_for_uniq = []
    for row in uniq:
        w_cnt, b_cnt = row[:5], row[5:]
        ws = "K" + "".join(letters[j] * int(w_cnt[j]) for j in range(5))
        bs = "K" + "".join(letters[j] * int(b_cnt[j]) for j in range(5))
        wval = int((w_cnt * vals).sum())
        bval = int((b_cnt * vals).sum())
        sig_for_uniq.append(f"{ws}v{bs}" if (wval, ws) >= (bval, bs)
                            else f"{bs}v{ws}")
    return np.array(sig_for_uniq)[inv]


def piece_counts(ds) -> np.ndarray:
    # Piece total per position (all pieces + 2 kings)
    wv_w, bv_b, wc, bc = _counts(ds)
    total = np.full(len(ds.eg_feats), 2, dtype=np.int32)  # two kings
    for stem, _, _ in _PIECES:
        total += wc[stem].astype(np.int32) + bc[stem].astype(np.int32)
    return total


def _three_class(model, X, true_cls, idx):
    #3-class accuracy + per-class recall + majority baseline, using the ordinal model's threshold-counting verdict on the given index set
    pred = T.coral_predict_class(T.ordinal_probs(model, X, idx))
    true = true_cls[idx]
    three = float((pred == true).mean())
    recalls = {}
    for c, name in enumerate(("loss", "draw", "win")):
        m = true == c
        recalls[name] = float((pred[m] == c).mean()) if m.any() else float("nan")
    counts = np.bincount(true, minlength=3)
    return three, recalls, counts.max() / max(1, counts.sum())

def confusion_at(model, X, y, idx, t_draw, t_win, tag="",
                 plot_out=None, normalize="none"):
    probs = T.ordinal_probs(model, X, idx)
    pred = T.coral_predict_class(probs, t_draw, t_win)
    true = T.label_to_class(y[idx])
    names = ("loss", "draw", "win")
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    print(f"\n{tag}  @ T_D={t_draw}, T_W={t_win}  (n={len(idx):,})")
    print(f"  overall acc: {(pred == true).mean():.3f}")
    for c, name in enumerate(names):
        m = true == c
        rec = (pred[m] == c).mean() if m.any() else float("nan")
        print(f"  recall {name:<5}: {rec:.3f}")
    for c, name in enumerate(names):
        print(f"  act {name:<5} " + " ".join(f"{cm[c, j]:>7,}" for j in range(3)))
    if plot_out:
        plot_confusion_matrix(
            cm, out=plot_out, normalize=normalize,
            title=f"Matrice de confuzie WDL  (T_D={t_draw}, T_W={t_win})",
            subtitle=f"{tag}n={len(idx):,}")
    return cm

def _ablation_acc(model, X, y, idx, mask_idx=-1):
    # 3-class accuracy with optional mean-imputation of feature mask_idx
    import torch
    Xa = X[idx].astype(np.float32).copy()
    if mask_idx >= 0:
        Xa[:, mask_idx] = float(model.feat_mean[mask_idx].item())  # neutral value
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xa).to(T.DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()       # (n, K-1)
    pred = T.coral_predict_class(probs)                   # threshold-count verdict
    true = T.label_to_class(y[idx])
    return float((pred == true).mean())


def feature_ablation(model, X, y, idx, eg_names, batch):
    # Ablate each feature by mean-imputation; rank by 3-CLASS accuracy drop
    base = _ablation_acc(model, X, y, idx)
    print(f"\n-- Feature ablation (3-class baseline acc = {base:.3f}, "
          f"n={len(idx):,}) --")
    print(f"  {'feature':<28}{'delta acc':>11}")
    drops = []
    for i, name in enumerate(eg_names):
        acc = _ablation_acc(model, X, y, idx, mask_idx=i)
        drops.append((name, base - acc))
    for name, drop in sorted(drops, key=lambda t: t[1], reverse=True):
        bar = "#" * max(0, int(drop * 300))
        print(f"  {name:<28}{drop:>+11.4f}  {bar}")


def linear_3class(ds, tr_idx, val_idx):
    # 3-class accuracy of a LINEAR (logistic) baseline on the geometry features -- the classical-eval ceiling
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
    except ImportError:
        return None
    # 3-class target: 0 loss, 1 draw, 2 win (same mapping as label_to_class)
    y3 = T.label_to_class(ds.labels)
    sc = StandardScaler()
    # multinomial is the default for multi-class targets in modern sklearn;
    # the explicit multi_class kwarg was removed in 1.7+.
    clf = LogisticRegression(max_iter=2000)
    Xtr = sc.fit_transform(ds.eg_feats[tr_idx])
    clf.fit(Xtr, y3[tr_idx])
    pred = clf.predict(sc.transform(ds.eg_feats[val_idx]))
    return accuracy_score(y3[val_idx], pred)


def _train(ds, tr_idx, val_idx, cfg, tag):
    #Return (model, true_cls)
    import torch
    true_cls = T.label_to_class(ds.labels)
    if cfg.load:
        dim = ds.eg_feats.shape[1]
        model = T.EndgameOrdinalNet(torch.zeros(dim), torch.ones(dim)).to(T.DEVICE)
        model.load_state_dict(torch.load(cfg.load, map_location=T.DEVICE))
        model.eval()
        print(f"  loaded checkpoint: {cfg.load}  (skipping training)")
        return model, true_cls
    cum = T.class_to_cumulative(true_cls)
    model = T.train_ordinal(ds.eg_feats, cum, true_cls, tr_idx, val_idx,
                            cfg.batch, cfg.epochs, cfg.lr, tag=tag)
    return model, true_cls

def mode_balanced(ds, cfg):
    tr_idx, val_idx = T.group_split(ds.game_ids, cfg.val_frac)
    diff, _, _ = material_value_diff(ds)
    bal_val = val_idx[diff[val_idx] <= cfg.threshold]
    print(f"  validation positions     : {len(val_idx):,}")
    print(f"  balanced (|dV| <= {cfg.threshold:g})   : {len(bal_val):,} "
          f"({len(bal_val)/max(1,len(val_idx)):.0%} of val)")
    sig_all = signatures(ds)
    sig = collections.Counter(sig_all[i] for i in bal_val)
    print("  top balanced configs     : "
          + ", ".join(f"{s}:{c}" for s, c in sig.most_common(8)))

    model, true_cls = _train(ds, tr_idx, val_idx, cfg, tag="geometry (full)")

    full_three, _, _ = _three_class(model, ds.eg_feats, true_cls, val_idx)
    bal_three, bal_rec, bal_maj = _three_class(model, ds.eg_feats, true_cls, bal_val)
    print("\n" + "=" * 60)
    print("FULL vs BALANCED (3-class accuracy)")
    print("=" * 60)
    print(f"  geometry MLP   full={full_three:.3f}   balanced={bal_three:.3f}")
    print(f"  majority baseline on balanced          : {bal_maj:.3f}  "
          f"(the honest floor the model must beat)")
    print(f"  -> MLP - majority on balanced          : {bal_three - bal_maj:+.3f}")
    print(f"  balanced recall  loss={bal_rec['loss']:.3f}  "
          f"draw={bal_rec['draw']:.3f}  win={bal_rec['win']:.3f}")
    lin = linear_3class(ds, tr_idx, bal_val) if cfg.linear else None
    if lin is not None:
        print(f"  linear (3-class) baseline on balanced  : {lin:.3f}  "
              f"(classical-eval ceiling)")
        print(f"  -> MLP - linear on balanced            : {bal_three - lin:+.3f}  "
              f"(value the nonlinear geometry adds where material cannot decide)")

    T.run_detailed_report(model, ds.eg_feats, true_cls, bal_val,
                          tag="geometry MLP [balanced subset]")

    # Deployed operating point (headline figure) + default 0.5/0.5 (appendix
    # companion that motivates the threshold sweep). Both on the balanced subset.
    confusion_at(model, ds.eg_feats, ds.labels, bal_val,
                 cfg.t_draw, cfg.t_win, tag="Număr poziții ",
                 plot_out=cfg.plot_out, normalize=cfg.plot_normalize)
    if cfg.plot_out:
        default_out = cfg.plot_out.replace(".pdf", "_default.pdf")
        if default_out == cfg.plot_out:
            default_out = cfg.plot_out + "_default.pdf"
    else:
        default_out = None
    confusion_at(model, ds.eg_feats, ds.labels, bal_val,
                 0.5, 0.5, tag="Număr poziții ",
                 plot_out=default_out, normalize=cfg.plot_normalize)

    feature_ablation(model, ds.eg_feats, ds.labels, bal_val, ds.eg_names, cfg.batch)


def mode_generalization(ds, probe, cfg):
    tr_idx, val_idx = T.group_split(ds.game_ids, cfg.val_frac)
    if probe.eg_feats.shape[1] != ds.eg_feats.shape[1]:
        raise SystemExit("feature width mismatch between train and probe sets")
    model, true_cls = _train(ds, tr_idx, val_idx, cfg, tag="geometry (train regime)")
    probe_cls = T.label_to_class(probe.labels)

    p_idx = np.arange(len(probe.labels))
    pc = piece_counts(probe)
    lo, hi = int(pc.min()), int(pc.max())
    sig = collections.Counter(signatures(probe))
    print(f"\nProbe: {len(probe.labels):,} positions, {lo}-{hi} men")
    print("  top probe configs: "
          + ", ".join(f"{s}:{c}" for s, c in sig.most_common(8)))
    cls = T.label_to_class(probe.labels)
    cb = np.bincount(cls, minlength=3)
    print(f"  probe label balance: loss {cb[0]/cb.sum():.0%} / "
          f"draw {cb[1]/cb.sum():.0%} / win {cb[2]/cb.sum():.0%}")

    T.run_detailed_report(model, ds.eg_feats, true_cls, val_idx,
                          tag="IN-REGIME validation")
    T.run_detailed_report(model, probe.eg_feats, probe_cls, p_idx,
                          tag="PROBE (extrapolation)")

    v3, vrec, vmaj = _three_class(model, ds.eg_feats, true_cls, val_idx)
    p3, prec, pmaj = _three_class(model, probe.eg_feats, probe_cls, p_idx)
    print("\n" + "=" * 60)
    print("GENERALIZATION SLOPE  (in-regime val -> probe)")
    print("=" * 60)
    print(f"  {'metric':<20}{'in-regime':>12}{'probe':>12}{'delta':>10}")
    print(f"  {'3-class accuracy':<20}{v3:>12.3f}{p3:>12.3f}{p3-v3:>+10.3f}")
    print(f"  {'  vs majority base':<20}{vmaj:>12.3f}{pmaj:>12.3f}")
    for name in ("loss", "draw", "win"):
        print(f"  {'recall '+name:<20}{vrec[name]:>12.3f}{prec[name]:>12.3f}"
              f"{prec[name]-vrec[name]:>+10.3f}")
    print("\n  Near-equal accuracy -> knowledge carries past the training")
    print("  frontier. Accuracy collapsing toward the majority baseline ->")
    print("  the model does not extrapolate to this regime.")


def mode_ablation(ds, cfg):
    tr_idx, val_idx = T.group_split(ds.game_ids, cfg.val_frac)
    model, true_cls = _train(ds, tr_idx, val_idx, cfg, tag="geometry")
    T.run_detailed_report(model, ds.eg_feats, true_cls, val_idx, tag="geometry MLP")
    feature_ablation(model, ds.eg_feats, ds.labels, val_idx, ds.eg_names, cfg.batch)


def mode_subset(ds, cfg):
    # Judge a SPARSE feature on the positions where it actually fires, instead of on the global average (which dilutes a 1%-firing feature to nothing)
    if cfg.feature is None:
        raise SystemExit("subset mode requires --feature NAME "
                        "(e.g. --feature bishop_color_stm)")
    if cfg.feature not in ds.eg_names:
        raise SystemExit(f"unknown feature '{cfg.feature}'. Available:\n  "
                        + ", ".join(ds.eg_names))
    fidx = ds.eg_names.index(cfg.feature)

    tr_idx, val_idx = T.group_split(ds.game_ids, cfg.val_frac)
    model, true_cls = _train(ds, tr_idx, val_idx, cfg, tag="geometry")

    # "fires" = differs from the feature's neutral/default value for features that is != 0; bishop_color uses -1 as its not-applicable sentinel, so firing means != -1.
    col = ds.eg_feats[:, fidx]
    default = -1.0 if "bishop_color" in cfg.feature else 0.0
    fires = col != default
    fire_val = val_idx[fires[val_idx]]

    print(f"\n-- Subset evaluation: '{cfg.feature}' --")
    print(f"  fires on {int(fires.sum()):,}/{len(col):,} positions "
          f"({fires.mean():.1%} of data); "
          f"{len(fire_val):,} in the validation set")
    if len(fire_val) < 50:
        print("  [warn] very few firing positions -- result is noisy")
    if len(fire_val) == 0:
        return

    base = _ablation_acc(model, ds.eg_feats, ds.labels, fire_val)
    masked = _ablation_acc(model, ds.eg_feats, ds.labels, fire_val, mask_idx=fidx)
    print(f"  3-class accuracy on firing subset:")
    print(f"    with feature      : {base:.3f}")
    print(f"    feature ablated    : {masked:.3f}")
    print(f"    -> contribution    : {base - masked:+.4f}  "
          f"(value the feature adds where it fires)")
    # For comparison, its global contribution on all validation positions.
    g_base = _ablation_acc(model, ds.eg_feats, ds.labels, val_idx)
    g_mask = _ablation_acc(model, ds.eg_feats, ds.labels, val_idx, mask_idx=fidx)
    print(f"  global contribution (all val): {g_base - g_mask:+.4f}  "
          f"(diluted across non-firing positions)")


def mode_coverage(ds, cfg):
    sig = signatures(ds)
    cls = T.label_to_class(ds.labels)
    pc = piece_counts(ds)
    print(f"\n-- Coverage: {len(ds.labels):,} positions, "
          f"{len(np.unique(sig))} distinct configs, {pc.min()}-{pc.max()} men --")
    counter = collections.Counter(sig)
    print(f"  {'config':<14}{'n':>8}{'loss':>7}{'draw':>7}{'win':>7}")
    for s, n in counter.most_common(cfg.top):
        m = sig == s
        c = np.bincount(cls[m], minlength=3)
        tot = c.sum()
        print(f"  {s:<14}{n:>8,}{c[0]/tot:>7.0%}{c[1]/tot:>7.0%}{c[2]/tot:>7.0%}")
    tail = len(counter) - cfg.top
    if tail > 0:
        print(f"  ... and {tail} more configs")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl")
    p.add_argument("probe", nargs="?", default=None,
                   help="probe file (generalization mode only)")
    p.add_argument("--mode", required=True,
                   choices=["balanced", "generalization", "ablation",
                            "coverage", "subset"])
    p.add_argument("--feature", default=None,
                   help="subset mode: feature name to evaluate on its firing "
                        "positions (e.g. bishop_color_stm)")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="balanced mode: max |white-black| material value")
    p.add_argument("--t-draw", type=float, default=0.35,
                   dest="t_draw",
                   help="balanced mode: P(>=draw) decision threshold "
                        "(deployed engine value; default 0.35)")
    p.add_argument("--t-win", type=float, default=0.65,
                   dest="t_win",
                   help="balanced mode: P(>=win) decision threshold "
                        "(deployed engine value; default 0.65)")
    p.add_argument("--plot-out", default="",
                   dest="plot_out",
                   help="balanced mode: path to save the confusion-matrix "
                        "figure (e.g. confusion_balanced.pdf). Empty = no plot.")
    p.add_argument("--plot-normalize", choices=["none", "rows"], default="rows",
                   dest="plot_normalize",
                   help="balanced mode: 'rows' shows per-actual-class "
                        "percentages (recommended for the thesis figure)")
    p.add_argument("--linear", action="store_true",
                   help="balanced mode: also score a logistic baseline")
    p.add_argument("--top", type=int, default=25,
                   help="coverage mode: how many configs to list")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--load", default="",
                   help="load a .pt checkpoint and analyze it instead of "
                        "training a fresh model (reflects the deployed net)")
    cfg = p.parse_args(argv)

    print(f"Loading {cfg.jsonl} ...")
    ds = T.EndgameDataset(cfg.jsonl, cfg.max_samples)
    print(f"  {len(ds.labels):,} positions, {ds.eg_feats.shape[1]} features")

    if cfg.mode == "coverage":
        mode_coverage(ds, cfg)
    elif cfg.mode == "ablation":
        mode_ablation(ds, cfg)
    elif cfg.mode == "subset":
        mode_subset(ds, cfg)
    elif cfg.mode == "balanced":
        mode_balanced(ds, cfg)
    elif cfg.mode == "generalization":
        if not cfg.probe:
            p.error("generalization mode requires a probe file argument")
        print(f"Loading probe {cfg.probe} ...")
        probe = T.EndgameDataset(cfg.probe, cfg.max_samples)
        mode_generalization(ds, probe, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()