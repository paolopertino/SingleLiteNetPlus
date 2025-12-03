def poly_lr_scheduler(max_epochs, initial_lr, optimizer, epoch, power=1.5):
    lr = round(initial_lr * (1 - epoch / max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
