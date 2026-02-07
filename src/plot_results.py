import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os


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
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    name = args.dataset_name.lower()

    data = json.load(open(os.path.join(".", "results", f"{name}.json"), "r"))
    plot_training_curves(
        data_dict=data,
        save_path=os.path.join(".", "results", f"{name}_training.png"),
        title=f"{name.upper()} Training Curves",
        xlabel="Epochs",
        ylabel="Error (%)",
    )
