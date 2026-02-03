import json
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    data_dict, save_path, title="Training Curves", xlabel="Epochs", ylabel="Error (%)"
):
    plt.figure(figsize=(10, 6))

    for legend_name, data in data_dict.items():
        if "test" in legend_name.lower():
            linestyle = "--"
            marker = "o"
        else:
            linestyle = "-"
            marker = "s"

        epochs = (np.arange(len(data)) + 1) * 10

        plt.plot(
            epochs,
            100 - np.array(data),
            label=legend_name,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=2,
            alpha=0.8,
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylim(0, 10)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.close()


if __name__ == "__main__":

    data = json.load(open("./results/cifar.json", "r"))
    plot_training_curves(
        data_dict=data,
        save_path="./results/cifar_training.png",
        title="CIFAR Training Curves",
        xlabel="Epoch",
        ylabel="Error (%)",
    )
