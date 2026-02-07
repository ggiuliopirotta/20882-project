import os
import torch
from torch.utils.data import ConcatDataset, random_split, Subset
from torchvision import datasets, transforms


class FlattenTransform:
    def __call__(self, x):
        return torch.flatten(x)


def load_dataset(dataset_name, root=None, download=True):
    if root is None:
        root = os.path.join(".", "data")

    dataset_class = getattr(datasets, dataset_name)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0), (1)),
            FlattenTransform(),
        ]
    )

    full_df = ConcatDataset(
        [
            dataset_class(
                root=root, train=split, download=download, transform=transform
            )
            for split in [True, False]
        ]
    )

    # Calculate sizes
    total_size = len(full_df)
    train_size = int(0.8 * total_size)
    dev_size = int(0.1 * total_size)
    test_size = total_size - train_size - dev_size

    # Split
    generator = torch.Generator()
    train_set, dev_set, test_set = random_split(
        full_df, [train_size, dev_size, test_size], generator=generator
    )
    train_set = Subset(train_set, range(1))

    print("Dataset loaded and split")
    print(f"- Train size: {len(train_set)}")
    print(f"- Dev size:   {len(dev_set)}")
    print(f"- Test size:  {len(test_set)}")
    print(f"- Feats:      {train_set[0][0].shape[0]}")

    return train_set, dev_set, test_set


if __name__ == "__main__":
    try:
        train, dev, test = load_dataset("MNIST")
    except Exception as e:
        print(f"Error: {e}")
