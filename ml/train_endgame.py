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

# 3-class encoding for the WDL target: 0=loss, 1=draw, 2=win.
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


def compute_norm(X_tr: np.ndarray):
    # Per-feature mean/std from TRAINING rows only; std floored to avoid /0
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # constant features -> pass through
    return (torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32))


def label_to_class(y: np.ndarray) -> np.ndarray:
    # {0.0, 0.5, 1.0} scalar WDL target -> ordinal class {0,1,2}
    cls = np.ones(len(y), dtype=np.int64)   # draw
    cls[y < 0.25] = 0                       # loss  (0.0)
    cls[y > 0.75] = 2                       # win   (1.0)
    return cls


def class_to_cumulative(c: np.ndarray, K: int = 3) -> np.ndarray:
    # class c -> (K-1) binary targets: t_k = 1 if c >= k+1
    # For K=3: t0 = [c>=1] (>=draw), t1 = [c>=2] (>=win)
    thresholds = np.arange(1, K)            # [1, 2]
    return (c[:, None] >= thresholds[None, :]).astype(np.float32)


class EndgameOrdinalNet(nn.Module):
    # MLP with a CORAL ordinal head (shared logit + K-1 ordered biases)
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, hidden=(64, 32), dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.register_buffer("feat_mean", mean)
        self.register_buffer("feat_std", std)
        self.num_thresholds = num_classes - 1   # K-1 = 2 WDL thresholds

        dims = [mean.numel(), *hidden]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        # Shared linear -> single logit (no bias here; the ordinal biases carry it)
        self.shared = nn.Linear(dims[-1], 1, bias=False)
        # Ordered thresholds, guaranteed b0 > b1 by construction (not just at init):
        #   b1 = bias_low                       (free)
        #   b0 = b1 + softplus(bias_gap) > b1   (softplus > 0 always)
        self.bias_low = nn.Parameter(torch.tensor(-2.0))
        self.bias_gap = nn.Parameter(torch.tensor(4.0))

    def ordered_biases(self) -> torch.Tensor:
        # Return [b0, b1] with b0 > b1 enforced. b0=P(>=draw) bias, b1=P(>=win).
        b1 = self.bias_low
        b0 = b1 + nn.functional.softplus(self.bias_gap)
        return torch.stack([b0, b1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns per-threshold LOGITS, shape (batch, K-1)
        # logits[:,0] -> P(>=draw), logits[:,1] -> P(>=win) after a sigmoid
        # Logits (not probabilities) are returned so the loss can use the
        # numerically stable binary_cross_entropy_with_logits; consumers that
        # need probabilities apply torch.sigmoid (or threshold logits at 0)
        x = (x - self.feat_mean) / self.feat_std
        logit = self.shared(self.trunk(x))          # (batch, 1)
        return logit + self.ordered_biases()        # broadcast -> (batch, K-1)


def coral_predict_class(probs: np.ndarray) -> np.ndarray:
    # Count thresholds passed at 0.5 -> class {0,1,2}.
    return (probs > 0.5).sum(axis=1).astype(int)


def _ordinal_loss(logits: torch.Tensor, cum_targets: torch.Tensor) -> torch.Tensor:
    # Summed binary cross-entropy over the K-1 thresholds, computed from LOGITS
    # via the numerically stable binary_cross_entropy_with_logits
    return nn.functional.binary_cross_entropy_with_logits(
        logits, cum_targets, reduction="mean") * logits.shape[1]


def _ord_loader(X, cum, idx, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X[idx].astype(np.float32)),
                    torch.from_numpy(cum[idx].astype(np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_ordinal(X, cum, true_cls, tr_idx, val_idx, batch, epochs, lr,
                tag="ordinal") -> EndgameOrdinalNet:
    mean, std = compute_norm(X[tr_idx])
    model = EndgameOrdinalNet(mean, std).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

    tr_loader = _ord_loader(X, cum, tr_idx, batch, True)

    print(f"\n-- Ordinal (CORAL) MLP training [{tag}]  [{DEVICE}]  "
        f"input_dim={X.shape[1]} --")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, tb in tr_loader:
            xb, tb = xb.to(DEVICE), tb.to(DEVICE)
            opt.zero_grad()
            loss = _ordinal_loss(model(xb), tb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        # validation: 3-class accuracy via threshold counting
        va = ordinal_accuracy(model, X, true_cls, val_idx)
        vl = running / len(tr_idx)
        sched.step(vl)
        print(f"  epoch {epoch:>2}/{epochs}  train={vl:.4f}  3cls_acc={va:.3f}  "
            f"biases={model.ordered_biases().detach().cpu().numpy().round(3)}")
    return model


def ordinal_probs(model, X, idx) -> np.ndarray:
    # Threshold probabilities (sigmoid of the model logits)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X[idx].astype(np.float32)).to(DEVICE))
        p = torch.sigmoid(logits)
    return p.cpu().numpy()


def ordinal_accuracy(model, X, true_cls, idx) -> float:
    pred = coral_predict_class(ordinal_probs(model, X, idx))
    return float((pred == true_cls[idx]).mean())


def run_detailed_report(model, X, true_cls, val_idx, tag="ordinal MLP") -> None:
    probs = ordinal_probs(model, X, val_idx)
    pred = coral_predict_class(probs)
    true = true_cls[val_idx]
    names = ("loss", "draw", "win")

    counts = np.bincount(true, minlength=3)
    n = counts.sum()
    majority = counts.max() / n
    overall = (pred == true).mean()

    print(f"\n-- Detailed report [{tag}] --")
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
        print(f"    {('act ' + name):>8}{row}")


def export_onnx(model, input_dim, eg_names, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    torch.onnx.export(
        model, torch.zeros(1, input_dim), path,
        input_names=["features"], output_names=["threshold_probs"],
        dynamic_axes={"features": {0: "batch"}, "threshold_probs": {0: "batch"}},
        opset_version=11, dynamo=False)
    norm = {"feature_order": eg_names,
            "mean": model.feat_mean.tolist(),
            "std": model.feat_std.tolist(),
            "thresh_bias": model.ordered_biases().detach().tolist()}
    npath = Path(path).with_suffix(".norm.json")
    npath.write_text(json.dumps(norm, indent=2))
    print(f"\n-- ONNX export -> {path}; normalization + thresh_bias written -> {npath}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--export", default="")
    p.add_argument("--load", default="")
    cfg = p.parse_args(argv)

    print(f"Loading {cfg.jsonl} ...")
    dataset = EndgameDataset(cfg.jsonl, cfg.max_samples)
    tr_idx, val_idx = group_split(dataset.game_ids, cfg.val_frac)
    true_cls = label_to_class(dataset.labels)
    cum = class_to_cumulative(true_cls)
    n_games = len(np.unique(dataset.game_ids))
    print(f"  {len(dataset.labels)} positions from {n_games} games  "
        f"|  train={len(tr_idx)}  val={len(val_idx)} (game-level split)")
    print(f"  class counts (loss/draw/win): {np.bincount(true_cls, minlength=3)}")

    if cfg.load:
        dim = dataset.eg_feats.shape[1]
        model = EndgameOrdinalNet(torch.zeros(dim), torch.ones(dim)).to(DEVICE)
        model.load_state_dict(torch.load(cfg.load, map_location=DEVICE))
    else:
        model = train_ordinal(dataset.eg_feats, cum, true_cls, tr_idx, val_idx, cfg.batch, cfg.epochs, cfg.lr)
        ckpt = (Path(cfg.export).with_suffix(".pt") if cfg.export
                else Path(cfg.jsonl).with_suffix(".ord.pt"))
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"  checkpoint saved -> {ckpt}")

    acc = ordinal_accuracy(model, dataset.eg_feats, true_cls, val_idx)
    print(f"\n  Ordinal MLP 3-class validation accuracy = {acc:.3f}")
    run_detailed_report(model, dataset.eg_feats, true_cls, val_idx)

    if cfg.export:
        export_onnx(model, dataset.eg_feats.shape[1], dataset.eg_names, cfg.export)
    print("\nDone.")


if __name__ == "__main__":
    main()