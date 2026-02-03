from dataset import load_dataset
from functions import train_unsupervised, train_supervised
import json
from models import KrotovHopfieldNetwork, FFNetwork
import torch
from torch.utils.data import Subset

torch.set_float32_matmul_precision("high")

# RTX 5060 Ti Optimizations
# - Auto-tune kernels
# - Use TF32 for matmul (faster on Ampere+)
# - Use TF32 for convolutions
# - Use TF32 Tensor Cores
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset_name = "CIFAR10"

    config = json.load(open("./src/config/cifar.json", "r"))

    train_set, dev_set, test_set = load_dataset(dataset_name)
    train_set = Subset(train_set, range(10))

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
        save_path = f"./results/{model_name}_cifar.pth"
        print(f"Starting training for {model_name} model...")

        if model_name == "ff":
            # Compile FF model for RTX 5060 Ti
            model = torch.compile(model, mode="max-autotune", fullgraph=False)

        if model_name == "kh":
            print("Training unsupervised...")
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

            torch.save(model.state_dict(), f"./results/{model_name}_unsup_cifar.pth")
            print(f"Model saved!")

            optimizer = torch.optim.Adam(model.S.parameters(), lr=0.001)
            print("Training supervised...")
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
            print("Training supervised...")
            train_acc, test_acc = train_supervised(
                model=model,
                train_loader=train_loader,
                device=device,
                optimizer=optimizer,
                m=config["m"],
                epochs=config["epochs_supervised"],
                test_loader=dev_loader,
            )

        torch.save(model.state_dict(), save_path)
        print(f"Model saved!")

        results_path = "./results/cifar.json"
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
