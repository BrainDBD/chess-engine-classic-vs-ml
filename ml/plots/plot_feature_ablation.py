import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import io

# Feature stem -> category. Stems are the feature name with any _stm/_opp
# suffix removed. Categories drive bar color and grouping.
_CATEGORY = {
    # material counts
    "pawns": "Caracteristică materială", "knights": "Caracteristică materială", "bishops": "Caracteristică materială",
    "rooks": "Caracteristică materială", "queens": "Caracteristică materială",
    # pawn / promotion geometry
    "passed_count": "Caracteristică geometrică", "connected_passers": "Caracteristică geometrică",
    "pawn_steps": "Caracteristică geometrică", "own_king_dist_promo": "Caracteristică geometrică",
    "enemy_king_dist_promo": "Caracteristică geometrică",
    "enemy_king_outside_square": "Caracteristică geometrică",
    "doubled": "Caracteristică geometrică", "isolated": "Caracteristică geometrică",
    "king_dist_own_pawn": "Caracteristică geometrică",
    # king geometry
    "king_dist": "Geometrie rege", "kings_collinear": "Geometrie rege",
    "king_gap_parity": "Geometrie rege", "king_edge_dist": "Geometrie rege",
    # other
    "bishop_color": "Alte", "opposite_bishops": "Alte",
    "rook_on_semiopen_file": "Alte", "rook_dist_enemy_passer": "Alte",
}

# category -> color (qualitative, colorblind-safe-ish)
_COLOR = {
    "Caracteristică materială": "#185FA5",          # blue
    "Caracteristică geometrică": "#1D9E75",   # teal
    "Geometrie rege": "#BA7517",    # amber
    "Alte": "#888780",              # gray
}
_CAT_ORDER = ["Caracteristică materială", "Caracteristică geometrică", "Geometrie rege", "Alte"]

# matches lines like '  queens_stm   +0.1931  ####'
_LINE = re.compile(r"^\s*([a-z][a-z_]+)\s+([+-]\d+\.\d+)")


def categorize(name: str) -> str:
    stem = name[:-4] if name.endswith(("_stm", "_opp")) else name
    return _CATEGORY.get(stem, "Alte")


def parse_ablation(path):
    raw = open(path, "rb").read()
    # detect BOM: UTF-16 LE/BE or UTF-8-SIG, else fall back to utf-8
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("utf-8", errors="replace")
    pairs = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            pairs.append((m.group(1), float(m.group(2))))
    if not pairs:
        raise SystemExit(f"no 'feature  +delta' lines found in {path}")
    return pairs


def _topn(pairs, n):
    return sorted(pairs, key=lambda t: t[1], reverse=True)[:n]


def plot_single(pairs, out, top, title, subtitle):
    rows = _topn(pairs, top)
    names = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    cats = [categorize(nm) for nm in names]
    colors = [_COLOR[c] for c in cats]

    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]   # largest at top
    ax.barh(y, deltas, color=colors, edgecolor="none")
    ax.set_xlim(0, max(deltas) * 1.15)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Δ acuratețe la eliminarea trăsăturii")
    ax.set_title(title or "Importanța trăsăturilor la acuratețea verdictului WDL")
    ax.axvline(0, color="0.6", linewidth=0.6)
    for yi, d in zip(y, deltas):
        ax.text(d + max(deltas) * 0.01, yi, f"{d:+.4f}", va="center",
                fontsize=8, color="0.25")
    # legend, only categories present
    present = [c for c in _CAT_ORDER if c in cats]
    handles = [plt.Rectangle((0, 0), 1, 1, color=_COLOR[c]) for c in present]
    ax.legend(handles, present, loc="lower right", fontsize=8, frameon=False)
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", va="bottom",
                 fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03 if subtitle else 0, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_paired(full_pairs, bal_pairs, out, top, title, subtitle):
    # Grouped bars: for the top-N features by FULL-set delta, show full vs balanced side by side. Designed to surface the material -> geometry inversion.
    bal = dict(bal_pairs)
    rows = _topn(full_pairs, top)
    names = [r[0] for r in rows]
    full_d = [r[1] for r in rows]
    bal_d = [bal.get(nm, 0.0) for nm in names]

    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.8))
    y = np.arange(len(rows))[::-1]
    h = 0.38
    ax.barh(y + h / 2, full_d, height=h, color="#185FA5", label="Set complet")
    ax.barh(y - h / 2, bal_d, height=h, color="#D85A30",
            label="Material echilibrat")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Δ acuratețe la eliminarea trăsăturii")
    ax.set_title(title or "Ablație: set complet vs. material echilibrat")
    ax.axvline(0, color="0.6", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", va="bottom",
                 fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03 if subtitle else 0, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Plot evaluate.py feature-ablation output as a tiered, "
                    "category-colored bar chart. Pass one file for a single "
                    "chart, or --balanced for a full-vs-balanced paired chart.")
    ap.add_argument("ablation", help="ablation dump (e.g. ablation_full.txt)")
    ap.add_argument("--balanced", default=None,
                    help="second ablation dump (material-balanced) -> paired chart")
    ap.add_argument("--out", default="feature_ablation.pdf")
    ap.add_argument("--top", type=int, default=15,
                    help="how many features to show (default 15)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle",
                    default="")
    args = ap.parse_args()

    full = parse_ablation(args.ablation)
    if args.balanced:
        bal = parse_ablation(args.balanced)
        plot_paired(full, bal, args.out, args.top, args.title, args.subtitle)
    else:
        plot_single(full, args.out, args.top, args.title, args.subtitle)


if __name__ == "__main__":
    main()