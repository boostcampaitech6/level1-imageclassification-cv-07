import argparse
import os
import multiprocessing
import random
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.mask_dataset import MaskDatasetV2
from models.mask_model import MaskModelV4
from utils.transform import TrainAugmentation, TestAugmentation
from utils.utils import get_lr
from ops.losses import get_loss

import warnings
warnings.filterwarnings('ignore')

_Optimizer = torch.optim.Optimizer
_Scheduler = torch.optim.lr_scheduler._LRScheduler


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(
    configs: Dict,
    dataloader: DataLoader,
    device: str, model: nn.Module,
    loss_fn: nn.Module,
    optimizer: _Optimizer,
    scheduler: _Scheduler,
    epoch: int
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
    accuracy_values = []

    epochs = configs['train']['epoch']

    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.float().to(device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value += loss.item()
        outputs = outputs > 0.5
        accuracy = (outputs == targets).float().mean().item()
        accuracy_values.append(accuracy)

        if (batch+1) % 50 == 0:
            train_loss = loss_value / 50
            train_acc = np.mean(accuracy_values)
            current_lr = get_lr(optimizer)
            image = images[0, ...].detach().cpu().numpy()
            image = image.transpose(1, 2, 0)

            print(
                f"Epoch[{epoch}/{epochs}]({batch + 1}/{len(dataloader)}) "
                f"| lr {current_lr} \ntrain loss {train_loss:4.4}"
                f" | train acc {train_acc}"
            )
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                'train_rgb': wandb.Image(image, caption='Input-image')
            })

            loss_value = 0
            train_acc = 0
            accuracy_values = []

    if scheduler is not None:
        scheduler.step()


def validation(
    dataloader: DataLoader,
    save_dir: os.PathLike,
    device: str,
    model: nn.Module,
    loss_fn: nn.Module,
    epoch: int
) -> None:
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
    num_batches = len(dataloader)
    model.eval()

    valid_loss = 0
    accuracy_values = []
    example_images = []

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.float().to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            valid_loss += loss.item()

            outputs = outputs > 0.5
            accuracy = (outputs == targets).float().mean().item()
            accuracy_values.append(accuracy)
            if batch % 50 == 0:
                outputs = str(outputs[0].cpu().numpy())
                targets = str(targets[0].cpu().numpy())
                example_images.append(wandb.Image(
                    images[0], caption="Pred: {} Truth: {}".format(outputs, targets)))

            wandb.log({"Image": example_images})

    valid_loss /= num_batches
    valid_acc = np.mean(accuracy_values)
    print(
        f"Epoch[{epoch}]({len(dataloader)})"
        f"valid loss {valid_loss:4.4} | valid acc {valid_acc:4.2%}"
    )
    wandb.log({
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
    })
    torch.save(
        model.state_dict(),
        f'{save_dir}/{epoch}-{valid_loss:4.4}-{valid_acc:4.2}.pth'
        )
    print(
        f'Saved Model State to {save_dir}/{epoch}-{valid_loss:4.4}-{valid_acc:4.2}.pth'
    )


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
    train_augmentation = TrainAugmentation(resize=[380, 380])
    train_data = MaskDatasetV2(
        image_dir=configs['data']['train_dir'],
        csv_path=configs['data']['csv_dir'],
        transform=train_augmentation,
        mode='train',
        valid_rate=configs['data']['valid_rate']
    )
    
    valid_augmentation = TestAugmentation(resize=[380, 380])
    val_data = MaskDatasetV2(
        image_dir=configs['data']['train_dir'],
        csv_path=configs['data']['csv_dir'],
        transform=valid_augmentation,
        mode='valid',
        valid_rate=configs['data']['valid_rate']
    )

    train_loader = DataLoader(
        train_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskModelV4().to(device)

    loss_fn = get_loss()
    optimizer = optim.Adam(model.parameters(), lr=configs['train']['lr'])
    scheduler = None

    save_dir = os.path.join(configs['ckpt_path'], str(model.__class__.__name__))
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

    for e in range(configs['train']['epoch']):
        print(f'Epoch {e+1}\n-------------------------------')
        train(
            configs, train_loader,
            device, model, loss_fn, optimizer, scheduler, e+1
        )
        validation(val_loader, save_dir, device, model, loss_fn, e+1)
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
