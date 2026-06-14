import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_endgame import EndgameDataset, group_split


# STM-relative WDL label -> bucket name. stm_result in {0.0, 0.5, 1.0}.
def label_bucket(stm_result: float) -> str:
    if stm_result < 0.25:
        return "lost"
    if stm_result > 0.75:
        return "won"
    return "draw"


def piece_count(fen: str) -> int:
    board_part = fen.split()[0]
    return sum(1 for c in board_part if c.isalpha())


# STM-relative material signature, e.g. "KRPvKR" means the side to move has
# K+R+P and the opponent has K+R. Pieces are ordered K,Q,R,B,N,P within a side.
# Because the buckets are STM-relative, the signature's left side is always the
# winning side in the 'won' bucket -- so "KQvKR" reads as STM(=winner) up Q-for-R.
_ORDER = "KQRBNP"


def material_signature(fen: str) -> str:
    parts = fen.split()
    board, stm = parts[0], parts[1]
    white = Counter(c for c in board if c.isupper())
    black = Counter(c for c in board if c.islower())

    def side(counter, upper: bool) -> str:
        s = ""
        for p in _ORDER:
            key = p if upper else p.lower()
            s += p * counter.get(key, 0)
        return s

    w, b = side(white, True), side(black, False)
    return f"{w}v{b}" if stm == "w" else f"{b}v{w}"


def _csv_set(s: str) -> set:
    return {x.strip() for x in s.split(",") if x.strip()}


def _apply_sig_filter(items, keep: set, drop: set):
    # items: list of (fen, signature). keep first (if non-empty), then drop.
    out = items
    if keep:
        out = [(f, s) for (f, s) in out if s in keep]
    if drop:
        out = [(f, s) for (f, s) in out if s not in drop]
    return out


def _cap(items, n: int, rng):
    if n > 0 and len(items) > n:
        idx = rng.permutation(len(items))[:n]
        return [items[j] for j in sorted(idx)]
    return items


def _print_breakdown(name, items, top=12):
    c = Counter(s for _, s in items)
    print(f"  {name} signatures ({len(c)} distinct, {len(items)} positions):")
    for sig, k in c.most_common(top):
        print(f"      {sig:<14} {k}")
    if len(c) > top:
        print(f"      ... and {len(c) - top} more")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("--outdir", default="test_epds")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="MUST match the value used in training (default 0.1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="MUST match group_split's seed used in training (default 42)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="MUST match the value used in training (0 = no cap)")
    ap.add_argument("--per-bucket", type=int, default=0,
                    help="cap positions per conversion/holding bucket (0 = no cap)")
    ap.add_argument("--max-pieces", type=int, default=6,
                    help="keep only positions with <= this many pieces (probe range)")
    # signature filters: scoped to won.epd / draw.epd (the conversion/holding suites).
    ap.add_argument("--won-signatures", default="",
                    help="comma-separated signatures to KEEP in won.epd "
                         "(e.g. KRPvKR,KQvKR). Empty = keep all.")
    ap.add_argument("--won-exclude", default="",
                    help="comma-separated signatures to DROP from won.epd "
                         "(e.g. KQvK,KRvK). Applied after --won-signatures.")
    ap.add_argument("--draw-signatures", default="",
                    help="like --won-signatures, for draw.epd (holding suite)")
    ap.add_argument("--draw-exclude", default="",
                    help="like --won-exclude, for draw.epd")
    cfg = ap.parse_args(argv)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.jsonl} (reproducing train/val split, seed={cfg.seed}) ...")
    ds = EndgameDataset(cfg.jsonl, cfg.max_samples)
    if (ds.game_ids == -1).all():
        sys.exit("[fatal] no usable game_id in dataset -- cannot guarantee a "
                 "leak-free split. Regenerate endgame.jsonl with game_id fields.")

    _, val_idx = group_split(ds.game_ids, cfg.val_frac, seed=cfg.seed)
    val_set = set(val_idx.tolist())
    print(f"  {len(ds.labels)} positions; {len(val_idx)} are in held-out (val) games")

    records = []
    with open(cfg.jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if cfg.max_samples and i >= cfg.max_samples:
                break
            records.append(json.loads(line))
    if len(records) != len(ds.labels):
        sys.exit(f"[fatal] row mismatch: {len(records)} jsonl lines vs "
                 f"{len(ds.labels)} dataset rows. Did the file change?")

    # collect held-out positions as (fen, signature), with leakage + piece filter
    buckets = {"won": [], "draw": [], "lost": []}
    wdl_rows = []   # (fen, stm_result, bucket, pieces, signature)
    kept = skipped_pieces = 0

    for i, rec in enumerate(records):
        if i not in val_set:
            continue                      # leakage guard: held-out games only
        fen = rec["fen"]
        if piece_count(fen) > cfg.max_pieces:
            skipped_pieces += 1
            continue
        stm = float(rec["syzygy"]["stm_result"]) if "syzygy" in rec \
            else float(rec["stm_wdl"])
        b = label_bucket(stm)
        sig = material_signature(fen)
        buckets[b].append((fen, sig))
        wdl_rows.append((fen, stm, b, piece_count(fen), sig))
        kept += 1

    print(f"  kept {kept} held-out positions (<= {cfg.max_pieces} pieces); "
          f"skipped {skipped_pieces} with too many pieces")
    print(f"  buckets: won={len(buckets['won'])}  draw={len(buckets['draw'])}  "
          f"lost={len(buckets['lost'])}")

    # show what's available so the user can choose signatures to keep/drop
    print("\n-- material-signature breakdown (STM-relative; winner's pieces first) --")
    _print_breakdown("won", buckets["won"])
    _print_breakdown("draw", buckets["draw"])

    rng = np.random.default_rng(cfg.seed)

    # conversion/holding suites: apply signature filters, then optional per-bucket cap
    won_keep, won_drop = _csv_set(cfg.won_signatures), _csv_set(cfg.won_exclude)
    draw_keep, draw_drop = _csv_set(cfg.draw_signatures), _csv_set(cfg.draw_exclude)

    won_out = _cap(_apply_sig_filter(buckets["won"], won_keep, won_drop),
                   cfg.per_bucket, rng)
    draw_out = _cap(_apply_sig_filter(buckets["draw"], draw_keep, draw_drop),
                    cfg.per_bucket, rng)
    lost_out = _cap(buckets["lost"], cfg.per_bucket, rng)

    written = {}
    for name, items in (("won", won_out), ("draw", draw_out), ("lost", lost_out)):
        if not items:
            continue
        fens = [f for f, _ in items]
        (outdir / f"{name}.epd").write_text("\n".join(fens) + "\n", encoding="utf-8")
        written[name] = len(fens)

    # balanced / mixed: FULL natural distribution (NOT signature-filtered), for the
    # headline strength comparison and fidelity context.
    full = {b: [f for f, _ in buckets[b]] for b in buckets}
    n_each = min(len(full["won"]), len(full["lost"]), len(full["draw"]))
    if n_each > 0:
        balanced = (full["won"][:n_each] + full["lost"][:n_each]
                    + full["draw"][:n_each])
        rng.shuffle(balanced)
        (outdir / "balanced.epd").write_text("\n".join(balanced) + "\n", encoding="utf-8")
        written["balanced"] = len(balanced)
    mixed = full["won"] + full["draw"] + full["lost"]
    rng.shuffle(mixed)
    if mixed:
        (outdir / "mixed.epd").write_text("\n".join(mixed) + "\n", encoding="utf-8")
        written["mixed"] = len(mixed)

    acc_path = outdir / "wdl_accuracy.jsonl"
    with acc_path.open("w", encoding="utf-8") as f:
        for fen, stm, b, pc, sig in wdl_rows:
            true_class = {"lost": 0, "draw": 1, "won": 2}[b]
            f.write(json.dumps({"fen": fen, "stm_result": stm,
                                "true_class": true_class, "pieces": pc,
                                "signature": sig}) + "\n")

    print(f"\nWrote to {outdir}/:")
    for name, n in written.items():
        tag = "  [signature-filtered]" if name in ("won", "draw") and (
            (name == "won" and (won_keep or won_drop)) or
            (name == "draw" and (draw_keep or draw_drop))) else ""
        print(f"  {name + '.epd':<18} {n} positions{tag}")
    print(f"  {'wdl_accuracy.jsonl':<18} {len(wdl_rows)} positions (true Syzygy labels)")


if __name__ == "__main__":
    main()