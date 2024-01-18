from torch.optim import lr_scheduler


def get_lambda_lr(optimizer, lr):
    return lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr,
        last_epoch=-1,
        verbose=False
    )


def get_step_lr(optimizer, step):
    return lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)


def get_cosine_lr(optimizer, t_max, eta_min):
    return lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=eta_min
    )


def get_cosine_restart(optimizer, t_0, t_mult, eta_min):
    return lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t_0,
        T_mult=t_mult,
        eta_min=0.00001
    )
