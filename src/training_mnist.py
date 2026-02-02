from dataset import load_dataset
from functions import train_unsupervised, train_supervised
import json
from models import KrotovHopfieldNetwork, FFNetwork
import torch
from torch.utils.data import Subset


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset_name = "MNIST"

    config = json.load(open("./src/config/mnist.json", "r"))

    train_set, dev_set, test_set = load_dataset(dataset_name)
    train_set = Subset(train_set, range(10))

    batch_size = config["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

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

    for model in [kh_model, ff_model]:
        model.to(device)

        model_name = "kh" if isinstance(model, KrotovHopfieldNetwork) else "ff"
        print(f"Starting training for {model_name} model...")

        if isinstance(model, KrotovHopfieldNetwork):

            train_unsupervised(
                model=model,
                train_loader=train_loader,
                device=device,
                p=config["p"],
                k=config["k"],
                delta=config["delta"],
                eps0=config["eps0"],
                precision=config["precision"],
                epochs=config["epochs_unsupervised"],
            )

            optimizer = torch.optim.Adam(model.S.parameters(), lr=0.001)
            train_acc, test_acc = train_supervised(
                model=model,
                train_loader=train_loader,
                device=device,
                optimizer=optimizer,
                m=config["m"],
                epochs=config["epochs_supervised"],
                test_loader=dev_loader,
            )

        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_acc, test_acc = train_supervised(
                model=model,
                train_loader=train_loader,
                device=device,
                optimizer=optimizer,
                m=config["m"],
                epochs=config["epochs_supervised"],
                test_loader=dev_loader,
            )

        results_path = "./results/mnist.json"
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {}

        results[model_name + "_train_acc"] = train_acc
        results[model_name + "_test_acc"] = test_acc

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print("Results saved!")
