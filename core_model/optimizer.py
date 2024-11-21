from torch import optim


def create_optimizer_scheduler(
    optimizer_type,
    parameters,
    epochs,
    learning_rate=1e-3,
    weight_decay=5e-4,
    eta_min=None,
    min_epochs_for_decay=20,
    factor=0.9
):
    # 根据用户选择的优化器初始化
    if optimizer_type == "adam":
        optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":  # add weight_decay, 0.7/0.8
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    if eta_min is None: eta_min = 0.01 * learning_rate
    if epochs >= min_epochs_for_decay:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
        # lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, max(1, epochs // 50), gamma=factor
        # )
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=factor
        )

    return optimizer, lr_scheduler
