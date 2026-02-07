import os
import torch
from torch.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity


def train_unsupervised(
    device,
    model,
    train_loader,
    p,
    k,
    delta,
    eps0,
    precision,
    epochs,
    use_profiler=False,
    profile_batches=0,
):
    model.to(device)
    profiler = None

    if use_profiler:
        if torch.cuda.is_available():
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
        else:
            print("cuda not available, profiler disabled")

    for epoch in range(epochs):

        eps = eps0 * (1 - epoch / epochs)
        # Profile on first epoch
        if epoch == 0 and profiler is not None:
            profiler.__enter__()

        for batch_idx, (bx, _) in enumerate(train_loader):
            v = bx.to(device, non_blocking=True)
            batch_size_range = torch.arange(v.size(0), device=device)

            with torch.no_grad(), autocast(device.type):
                # 1. Rank hidden units based on input currents
                currents = torch.matmul(
                    v, (torch.sign(model.W) * torch.abs(model.W) ** (p - 1)).T
                )
                ranked = torch.argsort(currents, dim=1)

                # 2. Create ranking activation 'g'
                yl = torch.zeros((model.W.shape[0], v.size(0)), device=device)
                yl[ranked[:, -1], batch_size_range] = 1.0
                yl[ranked[:, -k], batch_size_range] = -delta

                # 3. Calculate update 'ds' for the whole batch
                xx = torch.sum(yl * currents.t(), dim=1, keepdim=True)
                ds = torch.matmul(yl, v) - xx * model.W

                # 4. Max-normalize update
                nc = torch.max(torch.abs(ds))
                nc = torch.clamp(nc, min=precision)

                # 5. Apply update
                model.W.add_(eps * (ds / nc))

            # Profile only first N batches of first epoch
            if epoch == 0 and profiler is not None:
                if batch_idx >= profile_batches:
                    break

        # Stop profiler after first epoch
        if epoch == 0 and profiler is not None:
            profiler.__exit__(None, None, None)
            print("\n" + "=" * 17)
            print("PROFILING RESULTS")
            print("=" * 17)
            print(
                profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            )
            profiler.export_chrome_trace(
                os.path.join(".", "results", "trace_unsupervised.json")
            )
            print("\nTrace exported to file")

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
    device,
    model,
    train_loader,
    optimizer,
    m,
    epochs,
    dev_loader=None,
    use_profiler=False,
    profile_batches=0,
):
    model.to(device)
    train_acc = []
    test_acc = []
    scaler = GradScaler(device)
    profiler = None

    if use_profiler:
        if torch.cuda.is_available():
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
        else:
            print("cuda not available, profiler disabled")

    for epoch in range(epochs):
        adjust_lr(optimizer, epoch)
        model.train()
        correct = 0
        total = 0

        # Profile on first epoch
        if epoch == 0 and profiler is not None:
            profiler.__enter__()

        for batch_idx, (bx, by) in enumerate(train_loader):
            v = bx.to(device, non_blocking=True)
            y_true = torch.eye(10)[by].to(device) * 2 - 1

            optimizer.zero_grad()
            with autocast(device.type):
                y = model(v)
                loss = torch.sum(torch.abs(y - y_true) ** m) / v.size(0)

            with torch.no_grad():
                preds = model(v).argmax(dim=1)
                correct += (preds == by.to(device)).sum().item()
                total += by.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Profile only first N batches of first epoch
            if epoch == 0 and profiler is not None:
                if batch_idx >= profile_batches:
                    break

        # Stop profiler after first epoch
        if epoch == 0 and profiler is not None:
            profiler.__exit__(None, None, None)
            print("\n" + "=" * 17)
            print("PROFILING RESULTS")
            print("=" * 17)
            print(
                profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            )
            profiler.export_chrome_trace(
                os.path.join(".", "results", "trace_supervised.json")
            )
            print("\nTrace exported to file")

        if epoch % 10 == 0:
            acc = 100 * correct / total
            width = len(str(epochs - 1))
            print(f"Run epoch {epoch:0{width}d} -> Accuracy: {acc:.2f}%")
            train_acc.append(acc)
            if dev_loader is not None:
                acc = test_model(device, model, dev_loader)
                print(f"Test Accuracy: {acc:.2f}%")
                test_acc.append(acc)

    return train_acc, test_acc


def train_ff_network(
    device, model, train_loader, criterion, optimizer, epochs, dev_loader=None
):
    model.to(device)
    train_acc = []
    test_acc = []
    scaler = GradScaler(device)

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            with autocast(device.type):
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
            if dev_loader is not None:
                acc = test_model(device, model, dev_loader)
                print(f"Test Accuracy: {acc:.2f}%")
                test_acc.append(acc)

    return train_acc, test_acc


def test_model(device, model, eval_loader):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for tx, ty in eval_loader:
            v = tx.to(device)
            y_true = ty.to(device)

            y_pred = model(v).argmax(dim=1)
            correct += (y_pred == y_true).sum().item()
            total += y_true.size(0)

    accuracy = 100 * correct / total
    return accuracy
