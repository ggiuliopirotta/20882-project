from dataset import load_dataset
from functions import train_unsupervised, train_supervised, test_model
import itertools
import json
from models import KrotovHopfieldNetwork
import torch
from torch.utils.data import Subset


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset_name = "CIFAR10"

    config = json.load(open("./src/config/cifar.json", "r"))

    train_set, dev_set, test_set = load_dataset(dataset_name)
    train_set = Subset(train_set, range(10000))

    batch_size = config["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size)

    model = KrotovHopfieldNetwork(
        n_features=config["n_features"],
        n_hidden=config["n_hidden"],
        n_classes=config["n_classes"],
        n_power=config["n_power"],
        beta=config["beta"],
    )

    grid = {"p": [2, 3, 4], "k": [2, 3, 4]}

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = {}

    n_combs = len(combinations)
    print(f"\nStarting tuning with {n_combs} combinations...")

    width = len(str(n_combs))
    for i, comb in enumerate(combinations):
        print(
            f"\n-------- Combination {i+1:0{width}d}/{n_combs}: p={comb['p']}, k={comb['k']} --------"
        )

        # 2. Train Unsupervised
        train_unsupervised(
            model=model,
            train_loader=train_loader,
            device=device,
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
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            m=config["m"],
            epochs=config["epochs_supervised"],
            test_loader=dev_loader,
        )

        # 4. Final Evaluation
        results[str(comb)] = test_model(
            model=model, test_loader=dev_loader, device=device
        )

    best_combination = max(results, key=results.get)
    print(f"\nBest combination: {best_combination} -> {results[best_combination]:.2f}%")
    with open("./results/tuning_cifar.json", "w") as f:
        json.dump(results, f)
