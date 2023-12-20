import random
from typing import Tuple

import numpy as np

import torch
from torch import Tensor, nn


_Optimizer = torch.optim.Optimizer


def seed_everything(seed: int) -> None:
    """
    시드 고정 method

    :param seed: 시드
    :type seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer: _Optimizer) -> float:
    """
    optimizer 통해 lr 얻는 method

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :return: learning_rate
    :rtype: float
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def mixup_aug(
        input: Tensor, target: Tensor, alpha: float = 1.0
        ) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    mixup code

    :param input: 이미지 데이터
    :type input: Tensor
    :param target: 레이블 데이터
    :type target: Tensor
    :param alpha: 값 조정
    :type alpha: Tensor
    :return: 섞인 데이터와 모든 라벨
    :rtype: float
    """
    lambda_ = np.random.beta(alpha, alpha)

    batch_size = input.size(0)
    index = torch.randperm(batch_size)

    mixed_input = lambda_ * input + (1 - lambda_) * input[index, :]
    labels_a, labels_b = target, target[index]

    return mixed_input, labels_a, labels_b, lambda_


def mixuploss(
        criterion: nn.Module, pred: Tensor, labels_a: Tensor,
        labels_b: Tensor, lambda_: float
        ) -> float:
    """
    mixup loss code

    :param criterion: 손실 함수
    :type criterion: nn.Module
    :param pred: 손실 함수
    :type pred: nn.Module
    :param labels_a: 손실 함수
    :type labels_a: nn.Module
    :param labels_b: 손실 함수
    :type labels_b: nn.Module
    :param lambda_: 손실 함수
    :type lambda_: nn.Module
    :return: loss_value
    :rtype: float
    """
    return lambda_ * criterion(pred, labels_a) + \
        (1 - lambda_) * criterion(pred, labels_b)

def cutmix_aug(input, target):
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        Wa = W * cut_rat
        Ha = H * cut_rat
        cut_w = Wa.astype(np.int64)
        cut_h = Ha.astype(np.int64)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    lam = np.random.beta(512, 384)
    rand_index = torch.randperm(input.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    '''
    bbx1 = 0 
    bby1 = 0
    bbx2 = 256
    bby2 = 384
    '''
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, target_a, target_b, lam


def cutmixloss(criterion, pred, target_a, target_b, lam):
    return criterion(pred, target_a) * lam + criterion(pred, target_b) * (1. - lam)