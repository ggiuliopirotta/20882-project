import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def visualize_learned_features(
    W,
    n_show=25,
    filename="visualization.png",
    image_size=None,
):
    W = W.detach().cpu().numpy()
    indices = np.random.choice(W.shape[0], n_show, replace=False)

    # Compute image size (sqrt of n_feats) if not provided
    if image_size is None:
        n_feats = W.shape[1]
        side_len = int(np.sqrt(n_feats))
        if side_len * side_len == n_feats:
            image_size = (side_len, side_len)
        else:
            # FALLBACK: CHANGE BASED ON DATASET
            image_size = (32, 32, 3)

    # Set up plotting grid
    side = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(side, side, figsize=(8, 8))
    axes_flat = axes.flatten() if side > 1 else [axes]

    vmax = np.max(np.abs(W[indices]))
    for i, ax in enumerate(axes_flat):
        if i < len(indices):
            ax.imshow(
                W[indices[i]].reshape(image_size), cmap="bwr", vmin=-vmax, vmax=vmax
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    try:
        path = os.path.join(".", "results", "ff_cifar")
        data = torch.load(path + ".pth", map_location=torch.device("cpu"))
        W = data.get("_orig_mod.fc1.weight")

        visualize_learned_features(W, filename=path + "_feats.png")
    except Exception as e:
        print(f"Error: {e}")
