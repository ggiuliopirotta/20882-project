from dataset import load_dataset
from utils import train_ff_network, train_supervised, train_unsupervised
import json
from models import FFNetwork, KrotovHopfieldNetwork
import os
import torch


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

    kh_model = KrotovHopfieldNetwork(
        n_features=config["n_features"],
        n_hidden=config["n_hidden"],
        n_classes=config["n_classes"],
        n_power=config["n_power"],
        beta=config["beta"],
    )

    ff_model = FFNetwork(
        n_features=config["n_features"],
        n_hidden=config["n_hidden"],
        n_classes=config["n_classes"],
    )

    kh_model = kh_model.to(device)
    ff_model = ff_model.to(device)

    for model in [kh_model, ff_model]:
        model_name = "kh" if model.__class__ == KrotovHopfieldNetwork else "ff"
        save_path = os.path.join(".", "results", f"{data_name}_{model_name}")
        rel_path = None
        print(f"Using {model_name} model")

        if model_name == "kh":
            print("Starting UNSUPERVISED...")
            train_unsupervised(
                device=device,
                model=model,
                train_loader=train_loader,
                p=config["p"],
                k=config["k"],
                delta=config["delta"],
                eps0=config["eps0"],
                precision=config["precision"],
                epochs=config["epochs_unsupervised"],
            )

            path = save_path + "_unsupervised.pth"
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")

            optimizer = torch.optim.Adam(model.S.parameters(), lr=0.001)
            print("Starting SUPERVISED...")
            train_acc, test_acc = train_supervised(
                device=device,
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                m=config["m"],
                epochs=config["epochs_supervised"],
                dev_loader=dev_loader,
            )
            rel_path = "_supervised.pth"

        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            print("Starting SUPERVISED...")
            train_acc, test_acc = train_ff_network(
                device=device,
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=config["epochs_supervised"],
                dev_loader=dev_loader,
            )
            rel_path = "_baseline.pth"

        path = save_path + rel_path
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

        results_path = os.path.join(".", "results", f"{data_name}.json")
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {}

        results[model_name + "_train_acc"] = train_acc
        results[model_name + "_test_acc"] = test_acc

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_path}")
