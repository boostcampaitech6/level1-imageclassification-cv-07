import numpy as np
import torch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def mixup_aug(input, target, alpha=1.0):
    lambda_ = np.random.beta(alpha, alpha) # np.random.beta -> 두 개의 인자를 받아 베타분포를 따르는 랜덤수 뽑음
    
    batch_size = input.size(0)
    index = torch.randperm(batch_size) # 0 ~ (batch_size-1)의 랜덤한 정수 순열을 뽑음
    
    mixed_input = lambda_ * input + (1 - lambda_) * input[index, :]
    labels_a, labels_b = target, target[index]

    return mixed_input, labels_a, labels_b, lambda_


def mixuploss(criterion, pred, labels_a, labels_b, lambda_):
    return lambda_ * criterion(pred, labels_a) + (1 - lambda_) * criterion(pred, labels_b)
