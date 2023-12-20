import argparse
import os
import multiprocessing
import random
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
import wandb
from sklearn.metrics import f1_score, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from datasets.mask_datasets import MaskSplitByProfileDataset
from models.mask_model import SingleLabelModel
from utils.utils import mixup_aug, mixuploss, cutmix_aug, cutmixloss
from utils.utils import get_lr, seed_everything
from ops.losses import get_loss
from ops.optim import get_optim

import warnings
warnings.filterwarnings('ignore')

_Optimizer = torch.optim.Optimizer
_Scheduler = torch.optim.lr_scheduler._LRScheduler
scaler = GradScaler()


def train(
    configs: Dict, dataloader: DataLoader, device: str,
    model: nn.Module, loss_fn: nn.Module, optimizer: _Optimizer,
    scheduler: _Scheduler, epoch: int, mix: str
) -> None:
    """
    데이터셋으로 훈련

    :param dataloader: PyTorch DataLoader
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 손실 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    :param scheduler: 훈련에 사용되는 스케줄러
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param epoch: 현재 훈련되는 epoch
    :type epoch: int
    :param mixup: mixup 사용 여부
    :type mixup: bool
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
        if mix == 'mixup' and (batch + 1) % 3 == 0:
            images, labels_a, labels_b, lambda_ = mixup_aug(images, targets)
            with autocast():
                outputs = model(images)
                loss = mixuploss(
                    loss_fn, pred=outputs, labels_a=labels_a,
                    labels_b=labels_b, lambda_=lambda_
                )
        elif mix == 'cutmix' and (batch + 1) % 3 == 0:
            images, target_a, target_b, lambda_ = cutmix_aug(images, targets)
            with autocast():
                outputs = model(images)
                loss = cutmixloss(
                    loss_fn, pred=outputs, target_a=target_a,
                    target_b=target_b, lambda_=lambda_
                )
        else:
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value += loss.item()
        outputs = outputs.argmax(dim=-1)
        accuracy += (outputs == targets).sum().item()

        log_term = configs['train']['log_interval']
        if (batch+1) % log_term == 0:
            train_loss = loss_value / log_term
            train_acc = accuracy / configs['train']['batch_size'] / log_term
            current_lr = get_lr(optimizer)

            print(
                f"Epoch[{epoch}/{epochs}]({batch + 1}/{len(dataloader)}) "
                f"| lr {current_lr} \ntrain loss {train_loss:4.4}"
                f" | train acc {train_acc}"
            )

            loss_value = 0
            accuracy = 0

        if not configs['fast_train_mode']:
            wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    'train_rgb': wandb.Image(
                        images[0], caption=f'{targets[0]}'
                    )
                }, step=epoch)

    if scheduler is not None:
        scheduler.step()


def validation(
    dataloader: DataLoader, save_dir: os.PathLike, device: str,
    model: nn.Module, loss_fn: nn.Module, epoch: int
) -> float:
    """
    데이터셋으로 검증

    :param dataloader: PyTorch DataLoader
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 손실 함수
    :type loss_fn: nn.Module
    :param epoch: 현재 훈련되는 epoch
    :type epoch: int
    :return: valid_loss
    :rtype: float
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
            val_labels.extend(targets.cpu().numpy())
            val_preds.extend(outputs.cpu().numpy())

            if batch % configs['train']['log_interval'] == 0:
                idx = random.randint(0, outputs.size(0)-1)
                outputs = str(outputs[idx].cpu().numpy())
                targets = str(targets[idx].cpu().numpy())
                if not configs['fast_train_mode']:
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
    print(classification_report(y_true=val_labels, y_pred=val_preds))

    if not configs['fast_train_mode']:
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


def run_pytorch(configs: Dict) -> None:
    """
    학습 파이토치 파이프라인

    :param configs: 학습에 사용할 config
    :type configs: dict
    """

    if not configs['fast_train_mode']:
        wandb.init(
            project="level1-imageclassification-cv-07",
            entity='naver-ai-tech-cv07',
            config={
                'seed': configs['seed'],
                'model': configs['model'],
                'img_size': configs['data']['image_size'],
                'loss': configs['train']['loss'],
                'optim': configs['train']['optim'],
                'batch_size': configs['train']['batch_size'],
                'lr': configs['train']['lr'],
                'epoch': configs['train']['epoch'],
                'imagenet': configs['train']['imagenet'],
                'early_patience': configs['train']['early_patience'],
                'mix': configs['train']['mix'],
            }
        )

    dataset = MaskSplitByProfileDataset(
        root_folder=configs['data']['train_dir'],
        valid_rate=configs['data']['valid_rate'],
        csv_path=configs['data']['csv_dir']
    )

    width, height = map(int, configs['data']['image_size'].split(','))
    if configs['train']['imagenet']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = dataset.mean
        std = dataset.std

    train_transforms = A.Compose([
        A.Resize(width, height),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    valid_transforms = A.Compose([
        A.Resize(width, height),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    dataset.set_transform(train_transforms, valid_transforms)
    train_data, val_data = dataset.split_dataset()

    train_loader = DataLoader(
        train_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SingleLabelModel().to(device)

    loss_fn = get_loss(configs['train']['loss'])
    optimizer = get_optim(configs['train']['optim'], model, configs)
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
            else:
                i += 1
        else:
            save_dir = os.path.join(save_dir, version)
            os.makedirs(save_dir)
            break

    best_loss = 100
    cnt = 0
    epoch = configs['train']['epoch'] if not configs['fast_train_mode'] else 1
    for e in range(epoch):
        print(f'Epoch {e+1}\n-------------------------------')
        train(
            configs, train_loader, device, model, loss_fn,
            optimizer, scheduler, e+1, configs['train']['mix']
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
