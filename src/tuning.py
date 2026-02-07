from dataset import load_dataset
from utils import train_unsupervised, train_supervised, test_model
import itertools
import json
from models import KrotovHopfieldNetwork
import os
import torch
from torch.utils.data import Subset


if __name__ == "__main__":

    # SET DATASET NAME (MNIST or CIFAR10)
    data_name = "cifar10"
    print(f"Dataset selected: {data_name.upper()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = json.load(open(os.path.join(".", "config", f"{data_name}.json"), "r"))
    train_set, dev_set, test_set = load_dataset(data_name.upper())
    batch_size = config["batch_size"]

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = KrotovHopfieldNetwork(
        n_features=config["n_features"],
        n_hidden=config["n_hidden"],
        n_classes=config["n_classes"],
        n_power=config["n_power"],
        beta=config["beta"],
    )
    model = model.to(device)

    grid = {"p": [2, 3, 4], "k": [5, 6, 7]}

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = {}

    n_combs = len(combinations)
    print(f"\nStarting hyperparameters tuning...")

    width = len(str(n_combs))
    for i, comb in enumerate(combinations):
        print(
            f"\n-------- Combination {i+1:0{width}d}/{n_combs}: p={comb['p']}, k={comb['k']} --------"
        )

        # 2. Train Unsupervised
        train_unsupervised(
            device=device,
            model=model,
            train_loader=train_loader,
            p=comb["p"],
            k=comb["k"],
            delta=config["delta"],
            eps0=config["eps0"],
            precision=config["precision"],
            epochs=config["epochs_unsupervised"],
        )

        # 3. Train Supervised
        optimizer = torch.optim.Adam(model.S.parameters(), lr=0.001)
        train_supervised(
            device=device,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            m=config["m"],
            epochs=config["epochs_supervised"],
            dev_loader=dev_loader,
        )

        # 4. Final Evaluation
        accuracy = test_model(device=device, model=model, eval_loader=test_loader)
        results[str(comb)] = accuracy

    best_comb = max(results, key=results.get)
    print(f"\nBest combination: {best_comb} -> {results[best_comb]:.2f}%")
    with open(os.path.join(".", "results", f"{data_name}_tuning.json"), "w") as f:
        json.dump(results, f)
