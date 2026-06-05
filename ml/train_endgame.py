import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import chess

sys.path.insert(0, str(Path(__file__).parent))
from endgame_features import endgame_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Scalar prob -> 3-class for REPORTING ONLY (the model outputs a scalar):
#   p < lo -> loss, lo <= p <= hi -> draw, p > hi -> win
DRAW_BAND = (0.4, 0.6)

class EndgameDataset:
    #In-memory endgame dataset loaded from extract_dataset.py JSONL
    def __init__(self, path: str, max_samples: int = 0):
        eg, labels, gids = [], [], []
        self.eg_names = None
        missing_gid = 0

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                board = chess.Board(rec["fen"])

                eg_dict = endgame_features(board)
                eg.append(list(eg_dict.values()))
                if self.eg_names is None:
                    self.eg_names = list(eg_dict.keys())

                if "syzygy" in rec:
                    labels.append(float(rec["syzygy"]["stm_result"]))
                else:
                    labels.append(float(rec["stm_wdl"]))

                gid = rec.get("game_id", -1)
                if gid == -1:
                    missing_gid += 1
                gids.append(gid)

                if max_samples and (i + 1) >= max_samples:
                    break

        self.eg_feats = np.array(eg,     dtype=np.float32)
        self.labels   = np.array(labels, dtype=np.float32)
        self.game_ids = np.array(gids,   dtype=np.int64)
        self.missing_gid = missing_gid


def group_split(game_ids: np.ndarray, val_frac: float, seed: int = 42):
    #Partition indices so all positions of a game land on the same side
    unique = np.unique(game_ids)
    if len(unique) < 2 or (unique == -1).all():
        print("  [warn] no usable game_id -> falling back to POSITION-level "
            "split; regenerate the dataset for a leak-free game split.")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(game_ids))
        n_val = max(1, int(len(game_ids) * val_frac))
        return perm[n_val:], perm[:n_val]

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val_games = max(1, int(len(unique) * val_frac))
    val_games = set(unique[:n_val_games].tolist())
    is_val = np.fromiter((g in val_games for g in game_ids),
                        dtype=bool, count=len(game_ids))
    return np.where(~is_val)[0], np.where(is_val)[0]

class EndgameValueNet(nn.Module):
    #MLP value net with input normalization as non-trainable buffers.
    def __init__(self, mean: torch.Tensor, std: torch.Tensor,
                 hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        self.register_buffer("feat_mean", mean)
        self.register_buffer("feat_std", std)
        dims = [mean.numel(), *hidden]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(dims[-1], 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    # forward(): x_norm = (x - mean) / std, then dense layers -> win prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / self.feat_std
        return self.net(x).squeeze(-1)


def compute_norm(X_tr: np.ndarray):
    # Per-feature mean/std from TRAINING rows only; std floored to avoid /0
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # constant features -> pass through
    return (torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32))

def _loader(X, y, idx, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X[idx].astype(np.float32)),
                    torch.from_numpy(y[idx].astype(np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def _make_criterion(loss: str):
    if loss == "mse":
        return nn.MSELoss()
    return nn.BCELoss()


def train_mlp(X, y, tr_idx, val_idx, batch, epochs, lr, tag="geometry", loss="bce") -> EndgameValueNet:
    mean, std = compute_norm(X[tr_idx])
    model = EndgameValueNet(mean, std).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    crit = _make_criterion(loss)

    tr_loader  = _loader(X, y, tr_idx,  batch, True)
    val_loader = _loader(X, y, val_idx, batch, False)

    print(f"\n-- MLP training [{tag}]  [{DEVICE}]  input_dim={X.shape[1]}  loss={loss} --")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            batch_loss = crit(model(xb), yb)
            batch_loss.backward()
            opt.step()
            running += batch_loss.item() * len(xb)
        vl, va = eval_model(model, val_loader, loss=loss)
        sched.step(vl)
        print(f"  epoch {epoch:>2}/{epochs}  "
                f"train={running/len(tr_idx):.4f}  val={vl:.4f}  acc={va:.3f}")
    return model


def eval_model(model, loader, mask_idx: int = -1, loss: str = "bce"):
    model.eval()
    crit = _make_criterion(loss)
    tot_loss = tot_correct = tot = 0
    with torch.no_grad():
        for xb, yb in loader:
            if mask_idx >= 0:
                xb = xb.clone()
                xb[:, mask_idx] = model.feat_mean[mask_idx].item()  # neutral
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            tot_loss    += crit(preds, yb).item() * len(xb)
            tot_correct += ((preds > 0.5) == (yb > 0.5)).sum().item()
            tot         += len(xb)
    # (mean loss, win/not-win accuracy)
    return tot_loss / tot, tot_correct / tot


def mlp_preds(model, X, idx) -> np.ndarray:
    #Hard win/not-win predictions (threshold 0.5)
    return (mlp_probs(model, X, idx) > 0.5).astype(int)


def mlp_probs(model, X, idx) -> np.ndarray:
    # Raw scalar win-probabilities p in (0,1), side-to-move POV."""
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(X[idx].astype(np.float32)).to(DEVICE))
    return p.cpu().numpy()


def _to_3class(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map scalar scores in [0,1] to 0=loss / 1=draw / 2=win via a draw band."""
    cls = np.ones(len(values), dtype=int)   # default: draw
    cls[values < lo] = 0
    cls[values > hi] = 2
    return cls


def run_detailed_report(model, X, y, val_idx, tag="geometry MLP",
                        draw_band=DRAW_BAND) -> None:
    
    # - majority-class baseline (the trivial floor accuracy must beat)
    # - per-outcome accuracy (recall on true wins / draws / losses separately)
    # - 3x3 confusion matrix (rows = actual, cols = predicted)
    probs = mlp_probs(model, X, val_idx)
    pred  = _to_3class(probs, *draw_band)
    true  = _to_3class(y[val_idx], 0.5, 0.5)
    names = ("loss", "draw", "win")

    counts = np.bincount(true, minlength=3)
    n = counts.sum()
    majority = counts.max() / n
    overall = (pred == true).mean()

    print(f"\n-- Detailed report [{tag}]  "
        f"(draw band {draw_band[0]:.2f}-{draw_band[1]:.2f}) --")
    print(f"  majority-class baseline : {majority:.3f}  "
        f"(always predict '{names[counts.argmax()]}')")
    print(f"  overall 3-class accuracy: {overall:.3f}")

    print("  per-outcome accuracy (recall):")
    for c, name in enumerate(names):
        m = true == c
        acc = (pred[m] == c).mean() if m.any() else float("nan")
        print(f"    {name:<5}: {acc:.3f}   (n={int(m.sum()):,})")

    cm = np.zeros((3, 3), dtype=int)
    for t, pr in zip(true, pred):
        cm[t, pr] += 1
    print("  confusion matrix (rows=actual, cols=predicted):")
    print(f"    {'':>8}{'pred loss':>11}{'pred draw':>11}{'pred win':>11}")
    for c, name in enumerate(names):
        row = "".join(f"{cm[c, j]:>11,}" for j in range(3))
        print(f"    {('act '+name):>8}{row}")


def export_onnx(model, input_dim, eg_names, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    torch.onnx.export(
        model, torch.zeros(1, input_dim), path,
        input_names=["features"], output_names=["win_probability"],
        dynamic_axes={"features": {0: "batch"}, "win_probability": {0: "batch"}},
        opset_version=11, dynamo=False)
    # Dump normalization + feature order for a hand-written C++ forward pass
    norm = {"feature_order": eg_names,
            "mean": model.feat_mean.tolist(),
            "std": model.feat_std.tolist()}
    npath = Path(path).with_suffix(".norm.json")
    npath.write_text(json.dumps(norm, indent=2))
    kb = Path(path).stat().st_size // 1024
    print(f"\n-- ONNX export -> {path} ({kb} KB); normalization baked in")
    print(f"   feature order + mean/std also written -> {npath}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl")
    p.add_argument("--max-samples", type=int,   default=0)
    p.add_argument("--val-frac",    type=float, default=0.1)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch",       type=int,   default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--loss", choices=["bce", "mse"], default="bce",
                help="training loss: bce (default) or mse. MSE is a cheap "
                        "alternative that may improve draw separation by not "
                        "forcing the sigmoid hard onto the 0.5 draw target")
    p.add_argument("--export",      default="")
    p.add_argument("--load",        default="",
                help="load a .pt checkpoint instead of training")
    cfg = p.parse_args(argv)

    print(f"Loading {cfg.jsonl} ...")
    dataset = EndgameDataset(cfg.jsonl, cfg.max_samples)
    tr_idx, val_idx = group_split(dataset.game_ids, cfg.val_frac)
    n_games = len(np.unique(dataset.game_ids))
    nondraw = (dataset.labels != 0.5).mean()
    print(f"  {len(dataset.labels)} positions from {n_games} games  "
        f"|  train={len(tr_idx)}  val={len(val_idx)} (game-level split)")
    print(f"  input = {dataset.eg_feats.shape[1]} geometry features")
    print(f"  label mean = {dataset.labels.mean():.3f}  non-draw ~ {nondraw:.0%}")
    if dataset.missing_gid:
        print(f"  [warn] {dataset.missing_gid} positions lacked game_id")

    if cfg.load:
        print(f"\nLoading checkpoint: {cfg.load}")
        # feat_mean/feat_std are saved buffers in the checkpoint and will be
        # restored by load_state_dict, so we only need correctly-shaped
        # placeholders here -- computing real statistics would be wasted work
        dim = dataset.eg_feats.shape[1]
        placeholder = torch.zeros(dim), torch.ones(dim)
        model = EndgameValueNet(*placeholder).to(DEVICE)
        model.load_state_dict(torch.load(cfg.load, map_location=DEVICE))
    else:
        model = train_mlp(dataset.eg_feats, dataset.labels, tr_idx, val_idx,
                        cfg.batch, cfg.epochs, cfg.lr, tag="geometry",
                        loss=cfg.loss)
        ckpt = (Path(cfg.export).with_suffix(".pt") if cfg.export
                else Path(cfg.jsonl).with_suffix(".pt"))
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"  checkpoint saved -> {ckpt}")

    _, acc = eval_model(model, _loader(dataset.eg_feats, dataset.labels,
                                    val_idx, cfg.batch, False))
    print(f"\n  Geometry MLP validation accuracy = {acc:.3f}")
    run_detailed_report(model, dataset.eg_feats, dataset.labels, val_idx,
                        tag="geometry MLP")

    if cfg.export:
        export_onnx(model, dataset.eg_feats.shape[1], dataset.eg_names, cfg.export)

    print("\nDone.")


if __name__ == "__main__":
    main()