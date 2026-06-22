import numpy as np
import matplotlib.pyplot as plt

# King PSTs copied from the engine
KING_MG = np.array([
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14
])

KING_EG = np.array([
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
    10,  17,  23,  15,  20,  45,  44,  13,
    -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
])

def draw_pst(ax, pst, title, vmin, vmax):
    board = pst.reshape(8, 8)

    # origin='lower' means rank 1 is at the bottom, like a chessboard from White's perspective
    im = ax.imshow(board, origin='lower', vmin=vmin, vmax=vmax, cmap='Blues')

    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

    ax.set_xticks(range(8))
    ax.set_xticklabels(files)
    ax.set_yticks(range(8))
    ax.set_yticklabels(ranks)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Linie")
    ax.set_ylabel("Coloană")

    # Annotate each square with its PST value
    for r in range(8):
        for c in range(8):
            value = board[r, c]
            text_color = "white" if value < (vmin + vmax) / 2 else "black"
            ax.text(c, r, str(value), ha='center', va='center', fontsize=8, color=text_color)

    return im

def main():
    # Use a common scale so the two PSTs are visually comparable
    vmin = min(KING_MG.min(), KING_EG.min())
    vmax = max(KING_MG.max(), KING_EG.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    im1 = draw_pst(axes[0], KING_MG, "PST Rege - Poziție Normală", vmin, vmax)
    im2 = draw_pst(axes[1], KING_EG, "PST Rege - Poziție Finală", vmin, vmax)

    cbar = fig.colorbar(im2, ax=axes, shrink=0.85)
    cbar.set_label("Valoare PST")

    # fig.suptitle("Comparație Vizuală între tabelele de poziție pentru rege", fontsize=14)

    plt.savefig("king_pst_heatmaps.pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()