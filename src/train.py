import argparse
import os
import multiprocessing
import random
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
import wandb
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.datasets import MaskSplitByProfileDataset
from models.mask_model import SingleLabelModel
from utils.utils import get_lr, mixup_aug, mixuploss, cutmix_aug, cutmixloss
from ops.losses import get_focal_loss, get_cross_entropy_loss
from ops.optim import get_adam

import warnings
warnings.filterwarnings('ignore')

_Optimizer = torch.optim.Optimizer
_Scheduler = torch.optim.lr_scheduler._LRScheduler
scaler = torch.cuda.amp.GradScaler()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(
    configs: Dict,
    dataloader: DataLoader,
    device: str,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: _Optimizer,
    scheduler: _Scheduler,
    epoch: int,
    mixup: bool,
    cutmix: bool,
) -> None:
    """데이터셋으로 뉴럴 네트워크를 훈련합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
    model.train()

    loss_value = 0
    train_loss = 0
    train_acc = 0
    accuracy = 0

    epochs = configs['train']['epoch']
    for batch, (images, targets) in enumerate(dataloader):
        images = images.float().to(device)
        targets = targets.long().to(device)
        if mixup and (batch + 1) % 3 == 0:
            images, labels_a, labels_b, lambda_ = mixup_aug(images, targets)
            outputs = model(images)
            loss = mixuploss(
                loss_fn, pred=outputs, labels_a=labels_a,
                labels_b=labels_b, lambda_=lambda_
            )
        else:
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            
        if cutmix and (batch + 1) % 4 == 0:
            images, target_a, target_b, lam = cutmix_aug(images, targets)
            outputs = model(images)
            loss = cutmixloss(
                loss_fn, pred=outputs, target_a=target_a,
                target_b=target_b, lam=lam
            )
        else:
            outputs = model(images)
            loss = loss_fn(outputs, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_value += loss.item()
        outputs = outputs.argmax(dim=-1)
        accuracy += (outputs == targets).sum().item()

        if (batch+1) % 50 == 0:
            train_loss = loss_value / 50
            train_acc = accuracy / configs['train']['batch_size'] / 50
            current_lr = get_lr(optimizer)

            print(
                f"Epoch[{epoch}/{epochs}]({batch + 1}/{len(dataloader)}) "
                f"| lr {current_lr} \ntrain loss {train_loss:4.4}"
                f" | train acc {train_acc}"
            )

            loss_value = 0
            accuracy = 0
        wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                'train_rgb': wandb.Image(images[0], caption=f'{targets[0]}')
            }, step=epoch)
    if scheduler is not None:
        scheduler.step()


def validation(
    dataloader: DataLoader,
    save_dir: os.PathLike,
    device: str,
    model: nn.Module,
    loss_fn: nn.Module,
    epoch: int
) -> float:
    """데이터셋으로 뉴럴 네트워크의 성능을 검증합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """
    model.eval()

    valid_loss = []
    valid_acc = []
    val_labels = []
    val_preds = []
    example_images = []

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = images.float().to(device)
            targets = targets.long().to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            valid_loss.append(loss.item())

            outputs = outputs.argmax(dim=-1)
            val_acc_item = (outputs == targets).sum().item()
            valid_acc.append(val_acc_item)
            val_labels.extend(targets.cpu().numpy())  # labal,pred 저장
            val_preds.extend(outputs.cpu().numpy())
            if batch % 50 == 0:
                idx = random.randint(0, outputs.size(0)-1)
                outputs = str(outputs[idx].cpu().numpy())
                targets = str(targets[idx].cpu().numpy())
                example_images.append(wandb.Image(
                    images[idx],
                    caption="Pred: {} Truth: {}".format(outputs, targets)
                ))

    val_loss = np.sum(valid_loss) / len(dataloader)
    val_acc = np.sum(valid_acc) / len(dataloader.dataset)
    val_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')
    print(
        f"Epoch[{epoch}]({len(dataloader)})"
        f"valid loss {val_loss:4.4} | valid acc {val_acc:4.2%}"
        f"\nvalid f1 score {val_f1:.5}"
    )
    wandb.log({
        "Image": example_images,
        "valid_loss": val_loss,
        "valid_acc": val_acc,
        "val_f1_score": val_f1
    }, step=epoch)
    torch.save(
        model.state_dict(),
        f'{save_dir}/{epoch}-{val_loss:4.4}-{val_acc:4.2}.pth'
        )
    print(
        f'Saved Model to {save_dir}/{epoch}-{val_loss:4.4}-{val_acc:4.2}.pth'
    )
    return val_loss


def run_pytorch(configs) -> None:
    """학습 파이토치 파이프라인

    :param configs: 학습에 사용할 config들
    :type configs: dict
    """
    wandb.init(
        project="level1-imageclassification-cv-07",
        entity='naver-ai-tech-cv07',
        config={
            'seed': configs['seed'],
            'lr': configs['train']['lr'],
            'model': configs['model'],
            'epoch': configs['train']['epoch'],
            'batch_size': configs['train']['batch_size'],
            'img_size': configs['data']['image_size'],
            'val_rate': configs['data']['valid_rate']
        }
    )
    width, height = map(int, configs['data']['image_size'].split(','))
    # train_transforms = TrainAugmentation(resize=[width, height])
    import albumentations as A
    mean = [0.548, 0.504, 0.479]
    std = [0.237, 0.247, 0.246]
    train_transforms = A.Compose([
        A.Resize(width, height),
        A.Normalize(mean=mean, std=std),
        A.pytorch.ToTensorV2()
    ])
    valid_transforms = A.Compose([
        A.Resize(width, height),
        A.Normalize(mean=mean, std=std),
        A.pytorch.ToTensorV2()
    ])

    dataset = MaskSplitByProfileDataset(
        image_dir=configs['data']['train_dir'],
        csv_path=configs['data']['csv_dir'],
        valid_rate=configs['data']['valid_rate']
    )

    dataset.set_transform(train_transforms)
    train_data, val_data = dataset.split_dataset()

    train_loader = DataLoader(
        train_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=True,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SingleLabelModel().to(device)

    #loss_fn = get_cross_entropy_loss()
    loss_fn = get_focal_loss()
    optimizer = get_adam(model, configs)
    scheduler = None

    save_dir = os.path.join(configs['ckpt_path'], str(model.name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    i = 0
    while True:
        version = 'v' + str(i)
        if os.path.exists(os.path.join(save_dir, version)):
            if not os.listdir(os.path.join(save_dir, version)):
                save_dir = os.path.join(save_dir, version)
                break
            i += 1
            continue
        else:
            save_dir = os.path.join(save_dir, version)
            os.makedirs(save_dir)
            break

    best_loss = 100
    cnt = 0
    for e in range(configs['train']['epoch']):
        print(f'Epoch {e+1}\n-------------------------------')
        train(
            configs, train_loader, device, model, loss_fn,
            optimizer, scheduler, e+1, configs['train']['mixup'], configs['train']['cutmix']
        )
        val_loss = validation(
            val_loader, save_dir, device, model, loss_fn, e+1
        )
        if val_loss < best_loss:
            best_loss = val_loss
            cnt = 0
        else:
            cnt += 1
        if cnt == configs['train']['early_patience']:
            print('Early Stopping!')
            break
        print('\n')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    seed_everything(configs['seed'])
    run_pytorch(configs=configs)
