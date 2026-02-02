import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_learned_features(
    W: torch.Tensor, n_show: int = 25, filename: str = "visualization.png"
):
    W = W.detach().cpu().numpy()
    indices = np.random.choice(W.shape[0], n_show, replace=False)
    vmax = np.max(np.abs(W[indices]))

    side = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(side, side, figsize=(8, 8))

    axes_flat = axes.flat if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes_flat):
        if i < len(indices):
            ax.imshow(W[indices[i]].reshape(28, 28), cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    W = torch.randn(100, 28 * 28)
    visualize_learned_features(W, filename="test.png")
