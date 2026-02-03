import torch
from torch.cuda.amp import autocast, GradScaler


def train_unsupervised(
    model, train_loader, device, p, k, delta, eps0, precision, epochs
):
    model.to(device)

    for epoch in range(epochs):
        # Use a linear learning rate annealing
        eps = eps0 * (1 - epoch / epochs)

        for bx, _ in train_loader:
            v = bx.to(device, non_blocking=True)

            with torch.no_grad():
                # 1. Rank the hidden units based on input currents
                currents = torch.matmul(
                    v, (torch.sign(model.W) * torch.abs(model.W) ** (p - 1)).T
                )
                ranked = torch.argsort(currents, dim=1)

                # 2. Create the ranking activation 'g'
                yl = torch.zeros((model.W.shape[0], v.size(0)), device=device)
                yl[ranked[:, -1], torch.arange(v.size(0))] = 1.0
                yl[ranked[:, -k], torch.arange(v.size(0))] = -delta

                # 3. Calculate the update 'ds' for the whole batch at once
                xx = torch.sum(yl * currents.t(), dim=1, keepdim=True)
                ds = torch.matmul(yl, v) - xx * model.W

                # 4. Max-normalize the update for stability
                nc = torch.max(torch.abs(ds))
                if nc <= precision:
                    nc = precision

                # 5. Apply the update
                model.W += eps * (ds / nc)

        if epoch % 100 == 0:
            width = len(str(epochs - 1))
            print(f"Run epoch {epoch:0{width}d}")


def adjust_lr(optimizer, epoch):
    for pg in optimizer.param_groups:
        if epoch < 100:
            lr = 1e-3
        elif epoch < 150:
            lr = 1e-4
        elif epoch < 200:
            lr = 1e-5
        elif epoch < 250:
            lr = 1e-6
        else:
            lr = 1e-7
        pg["lr"] = lr


def train_supervised(
    model, train_loader, device, optimizer, m, epochs, test_loader=None
):
    model.to(device)
    train_acc = []
    test_acc = []
    scaler = GradScaler()

    for epoch in range(epochs):
        adjust_lr(optimizer, epoch)
        model.train()

        correct = 0
        total = 0

        for bx, by in train_loader:
            v = bx.to(device)
            y_true = torch.eye(10)[by].to(device) * 2 - 1

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                y = model(v)
                loss = torch.sum(torch.abs(y - y_true) ** m) / v.size(0)

            with torch.no_grad():
                preds = model(v).argmax(dim=1)
                correct += (preds == by.to(device)).sum().item()
                total += by.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 10 == 0:
            acc = 100 * correct / total
            width = len(str(epochs - 1))
            print(f"Run epoch {epoch:0{width}d} -> Accuracy: {acc:.2f}%")
            train_acc.append(acc)
            if test_loader is not None:
                acc = test_model(model, test_loader, device)
                print(f"Test Accuracy: {acc:.2f}%")
                test_acc.append(acc)

    return train_acc, test_acc


def train_ff_network(
    model, train_loader, optimizer, criterion, device, epochs, test_loader
):
    model.to(device)
    train_acc = []
    test_acc = []
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()

        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                y = model(bx)
                loss = criterion(y, by)

            with torch.no_grad():
                preds = y.argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 10 == 0:
            acc = 100 * correct / total
            width = len(str(epochs - 1))
            print(f"Run epoch {epoch:0{width}d} -> Accuracy: {acc:.2f}%")
            train_acc.append(acc)
            if test_loader is not None:
                acc = test_model(model, test_loader, device)
                print(f"Test Accuracy: {acc:.2f}%")
                test_acc.append(acc)

    return train_acc, test_acc


def test_model(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for tx, ty in test_loader:
            v = tx.to(device)
            y_true = ty.to(device)

            y_pred = model(v).argmax(dim=1)
            correct += (y_pred == y_true).sum().item()
            total += y_true.size(0)

    accuracy = 100 * correct / total
    return accuracy
