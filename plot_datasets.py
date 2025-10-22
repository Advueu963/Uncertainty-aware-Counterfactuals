import matplotlib.pyplot as plt
import seaborn as sns
from MasterThesis.src.uncertainty_cfs.data import (
    load_bubbles,
    load_bubbles_noisy,
    load_infinity_dataset,
    load_l_dataset,
    load_one_moon,
    load_ring_dataset,
    load_two_moon,
)

sns.set(style="white", palette="muted", font_scale=1.3)


def plot_datasets():
    dataset_loaders = [
        ("Bubbles", load_bubbles, {"n_samples": 1000}),
        ("L Dataset", load_l_dataset, {"n_samples": 333}),
        ("One Moon", load_one_moon, {"n_samples": 1000}),
        (
            "Ring",
            load_ring_dataset,
            {"n_samples": 1000, "inner_radius": 1.0, "outer_radius": 2.0, "noise": 0.1},
        ),
        ("Bubbles Noisy", load_bubbles_noisy, {"n_samples": 1000}),
        ("Two Moon", load_two_moon, {"n_samples": 1000}),
        ("Infinity", load_infinity_dataset, {"n_samples": 1000}),
    ]

    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(hspace=0.4)

    # First row: 1x4 grid
    axs_row1 = [fig.add_subplot(2, 4, i + 1) for i in range(4)]

    colors = sns.color_palette("Set1", 2)

    for ax, (name, loader, params) in zip(axs_row1, dataset_loaders[:4]):
        X, y, _ = loader(**params)
        for cls in [0, 1]:
            ax.scatter(
                X[y == cls, 0],
                X[y == cls, 1],
                s=8,
                color=colors[cls],
                label=f"Class {cls}",
                alpha=0.7,
            )
        ax.set_title(name, fontsize=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal")
        # ax.legend(loc='upper right', fontsize=10, frameon=False)

    # Second row: manually add 3 axes, centered
    lefts = [0.18, 0.40, 0.62]
    width = 0.18
    height = 0.35
    bottom = 0.08
    axs_row2 = []
    for i in range(3):
        ax = fig.add_axes([lefts[i], bottom, width, height])
        name, loader, params = dataset_loaders[4 + i]
        X, y, _ = loader(**params)
        for cls in [0, 1]:
            ax.scatter(
                X[y == cls, 0],
                X[y == cls, 1],
                s=8,
                color=colors[cls],
                label=f"Class {cls}",
                alpha=0.7,
            )
        ax.set_title(name, fontsize=16, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal")
        # ax.legend(loc='upper left', fontsize=10, frameon=False)
        axs_row2.append(ax)

    # Collect handles and labels from the first subplot (or all, if needed)
    handles, labels = axs_row1[0].get_legend_handles_labels()

    # Add a single legend to the figure
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))

    plt.suptitle("Synthetic Datasets", fontsize=22, fontweight="bold", y=0.97)
    plt.savefig("synthetic_datasets.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_datasets()
