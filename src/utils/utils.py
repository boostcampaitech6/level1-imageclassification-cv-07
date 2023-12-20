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


def cutmix_aug(input, target):
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        Wa = W * cut_rat #np.int was delet numpy 1.24
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
    
    lam = np.random.beta(512, 384) #hyperparameter (image size)
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