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
        inp: Tensor, target: Tensor, alpha: float = 1.0
        ) -> Tuple[Tensor, Tensor, Tensor, float]:
    """
    mixup code

    :param inp: 이미지 데이터
    :type inp: Tensor
    :param target: 레이블 데이터
    :type target: Tensor
    :param alpha: 값 조정
    :type alpha: Tensor
    :return: 섞인 이미지, 레이블, 섞인 이미지 레이블, 조정 값
    :rtype: Tuple[Tensor, Tensor, Tensor, float]
    """
    lambda_ = np.random.beta(alpha, alpha)

    batch_size = inp.size(0)
    index = torch.randperm(batch_size)

    mixed_input = lambda_ * inp + (1 - lambda_) * inp[index, :]
    labels_a, labels_b = target, target[index]

    return mixed_input, labels_a, labels_b, lambda_


def mixuploss(
        loss_fn: nn.Module, pred: Tensor, l_a: Tensor,
        l_b: Tensor, lambda_: float
        ) -> float:
    """
    mixup loss code

    :param loss_fn: 손실 함수
    :type loss_fn: nn.Module
    :param pred: 예측 레이블 값
    :type pred: Tensor
    :param l_a: 레이블
    :type l_a: Tensor
    :param l_b: 섞은 레이블
    :type l_b: Tensor
    :param lambda_: 조정 값
    :type lambda_: float
    :return: loss_value
    :rtype: float
    """
    return lambda_ * loss_fn(pred, l_a) + (1 - lambda_) * loss_fn(pred, l_b)


def rand_bbox(size, lam) -> Tuple[int, int, int, int]:
    '''
    bbox random하게 뽑기
    0, 0, 384, 512가 format

    :return: bbx1, bby1, bbx2, bby2
    :rtype: Tuple[int, int, int, int]
    '''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    Wa = W * cut_rat
    Ha = H * cut_rat
    cut_w = Wa.astype(np.int64)
    cut_h = Ha.astype(np.int64)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_aug(inp: Tensor, target: Tensor):
    """
    cutmix code

    :param inp: 이미지 데이터
    :type inp: Tensor
    :param target: 레이블 데이터
    :type target: Tensor
    :return: 섞인 이미지, 레이블, 섞인 이미지 레이블, 조정 값
    :rtype: Tuple[Tensor, Tensor, Tensor, float]
    """
    lam = np.random.beta(inp.size()[2], inp.size()[3])
    rand_idx = torch.randperm(inp.size()[0]).cuda()
    target_a = target
    target_b = target[rand_idx]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inp.size(), lam)
    inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - (
        (bbx2 - bbx1) * (bby2 - bby1) / (inp.size()[-1] * inp.size()[-2])
    )

    return inp, target_a, target_b, lam


def cutmixloss(
        loss_fn: nn.Module, pred: Tensor, l_a: Tensor,
        l_b: Tensor, lambda_: float
        ) -> float:
    """
    cutmix loss code

    :param loss_fn: 손실 함수
    :type loss_fn: nn.Module
    :param pred: 예측 레이블 값
    :type pred: Tensor
    :param l_a: 레이블
    :type l_a: Tensor
    :param l_b: 섞은 레이블
    :type l_b: Tensor
    :param lambda_: 조정 값
    :type lambda_: float
    :return: loss_value
    :rtype: float
    """
    return loss_fn(pred, l_a) * lambda_ + loss_fn(pred, l_b) * (1. - lambda_)
