import argparse
import numpy as np
import matplotlib.pyplot as plt


# Fallback matrix for standalone use (e.g. re-plotting a saved result without
# re-running evaluate.py). evaluate.py passes its own freshly computed matrix.
DEFAULT_CM = np.array([
    [33649, 2132,    52],
    [ 2440, 21991, 1695],
    [  230, 2515, 35053],
], dtype=float)

CLASS_NAMES = ["Pierdere", "Remiză", "Câștig"]


def format_count(x: float) -> str:
    return f"{int(round(x)):,}"


def plot_confusion_matrix(cm, out="confusion_matrix_wdl.pdf",
                          normalize="none", dpi=200, title=None,
                          subtitle=None):
    cm = np.asarray(cm, dtype=float)

    if normalize == "rows":
        row_sums = cm.sum(axis=1, keepdims=True)
        values = np.divide(cm, row_sums, out=np.zeros_like(cm),
                           where=row_sums != 0) * 100.0
        head = title or "Matrice de confuzie WDL, normalizată pe rânduri"
        cbar_label = "Procent după clasa reală"
        vmin, vmax = 0.0, 100.0
        text_values = [[f"{values[i, j]:.1f}%\n({format_count(cm[i, j])})"
                        for j in range(cm.shape[1])]
                       for i in range(cm.shape[0])]
    else:
        values = cm
        head = title or "Matrice de confuzie WDL"
        cbar_label = "Număr de poziții"
        # Scale the colormap to the data, not a hardcoded 0-100. With counts in
        # the tens of thousands a fixed vmax=100 saturates every cell to the same
        # blue; deriving vmax from the matrix restores a readable gradient.
        vmin, vmax = 0.0, (values.max() if values.max() > 0 else 1.0)
        text_values = [[format_count(cm[i, j]) for j in range(cm.shape[1])]
                       for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    image = ax.imshow(values, cmap="Blues", vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(head)
    ax.set_xlabel("Clasa prezisă")
    ax.set_ylabel("Clasa reală")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = vmax / 2.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] > threshold else "black"
            ax.text(j, i, text_values[i][j], ha="center", va="center",
                    color=text_color)

    total = cm.sum()
    accuracy = np.trace(cm) / total if total else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        recalls = np.diag(cm) / cm.sum(axis=1)
    summary = (
        f"Acuratețe globală: {accuracy:.3f}   "
        f"Rata de identificare: pierdut: {recalls[0]:.3f}   "
        f"egal: {recalls[1]:.3f}   "
        f"victorie: {recalls[2]:.3f}"
    )
    bottom = 0.04
    fig.text(0.5, 0.015, summary, ha="center", va="bottom", fontsize=9)
    if subtitle:
        bottom = 0.075
        fig.text(0.5, 0.045, subtitle, ha="center", va="bottom", fontsize=8,
                 color="0.35")

    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="confusion_matrix_wdl.pdf",
                        help="Output image path.")
    parser.add_argument("--normalize", choices=["none", "rows"],
                        default="none",
                        help="Use 'rows' to show per-actual-class percentages.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    args = parser.parse_args()
    plot_confusion_matrix(DEFAULT_CM, out=args.out, normalize=args.normalize,
                          dpi=args.dpi)


if __name__ == "__main__":
    main()