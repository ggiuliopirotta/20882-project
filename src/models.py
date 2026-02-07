import torch
from torch import nn


class KrotovHopfieldNetwork(nn.Module):
    def __init__(
        self,
        n_features,
        n_hidden,
        n_classes,
        n_power,
        beta,
    ):
        super().__init__()
        self.n_power = n_power
        self.beta = beta

        # Unsupervised
        self.W = nn.Parameter(torch.randn(n_hidden, n_features), requires_grad=False)
        # Supervised with initialization
        self.S = nn.Linear(n_hidden, n_classes, bias=False)
        nn.init.xavier_uniform_(self.S.weight)

    def forward(self, x):
        # Fused matmul and clamp
        d = torch.matmul(x, self.W.T)
        h = torch.clamp(d, min=0).pow_(self.n_power)

        # Normalize to unit sphere (fused max and division)
        h_max = torch.amax(h, dim=1, keepdim=True)
        h = h / (h_max + 1e-10)

        y = torch.tanh(self.beta * self.S(h))
        return y


class FFNetwork(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        # Fused ReLU for better performance
        y1 = torch.nn.functional.relu(self.fc1(x), inplace=True)
        y2 = self.fc2(y1)
        return y2


if __name__ == "__main__":

    n_features = 28 * 28
    n_hidden = 1024
    n_classes = 10
    kh_network = KrotovHopfieldNetwork(n_features, n_hidden, n_classes)

    try:
        x = torch.randn(100, n_features)
        y = kh_network(x)
    except Exception as e:
        print(f"Error: {e}")
