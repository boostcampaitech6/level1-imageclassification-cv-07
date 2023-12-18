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
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.mask_dataset import SingleLabelDataset
from models.mask_model import SingleLabelModel
from utils.transform import TrainAugmentation, TestAugmentation
from utils.utils import get_lr
from ops.losses import get_cross_entropy_loss

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
    device: str,
    model: nn.Module,
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
    accuracy = 0

    epochs = configs['train']['epoch']

    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.long().to(device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                'train_rgb': wandb.Image(images[0], caption=f'{targets[0]}')
            })

            loss_value = 0
            accuracy = 0

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
    model.eval()

    valid_loss = []
    valid_acc = []
    val_labels = []
    val_preds = []
    example_images = []

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = images.to(device)
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
                idx = random.randint(0, len(outputs))
                outputs = str(outputs[idx].cpu().numpy())
                targets = str(targets[idx].cpu().numpy())
                example_images.append(wandb.Image(
                    images[idx], caption="Pred: {} Truth: {}".format(outputs, targets)))

            wandb.log({"Image": example_images})
    val_loss = np.sum(valid_loss) / len(dataloader)
    val_acc = np.sum(valid_acc) / len(dataloader.dataset)
    val_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')
    print(
        f"Epoch[{epoch}]({len(dataloader)})"
        f"valid loss {val_loss:4.4} | valid acc {val_acc:4.2%}"
        f"\nvalid f1 score {val_f1:.5}"
    )
    wandb.log({
        "valid_loss": val_loss,
        "valid_acc": val_acc,
        "val_f1_score": val_f1
    })
    torch.save(
        model.state_dict(),
        f'{save_dir}/{epoch}-{val_loss:4.4}-{val_acc:4.2}.pth'
        )
    print(
        f'Saved Model to {save_dir}/{epoch}-{val_loss:4.4}-{val_acc:4.2}.pth'
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
    train_data = SingleLabelDataset(
        image_dir=configs['data']['train_dir'],
        csv_path=configs['data']['csv_dir'],
        transform=train_augmentation,
        mode='train',
        valid_rate=configs['data']['valid_rate']
    )
    print(train_data[0])

    valid_augmentation = TestAugmentation(resize=[380, 380])
    val_data = SingleLabelDataset(
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

    loss_fn = get_cross_entropy_loss()
    optimizer = optim.Adam(model.parameters(), lr=configs['train']['lr'])
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