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

from datasets.mask_dataset import MultiLabelDataset
from models.mask_model import MultiLabelModel
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
    mask_loss_value = 0
    gender_loss_value = 0
    age_loss_value = 0

    mask_match = 0
    gender_match = 0
    age_match = 0

    epochs = configs['train']['epoch']

    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        mask_label, gender_label, age_label = targets
        mask_label = mask_label.long().to(device)
        gender_label = gender_label.long().to(device)
        age_label = age_label.long().to(device)

        mask_out, gen_out, age_out = model(images)
        mask_loss = loss_fn(mask_out, mask_label)
        gender_loss = loss_fn(gen_out, gender_label)
        age_loss = loss_fn(age_out, age_label)
        total_loss = mask_loss + gender_loss + age_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        mask_loss_value += mask_loss.item()
        gender_loss_value += gender_loss.item()
        age_loss_value += age_loss.item()
        loss_value += total_loss.item()

        mask_match += (mask_out.argmax(dim=-1) == mask_label).sum().item()
        gender_match += (gen_out.argmax(dim=-1) == gender_label).sum().item()
        age_match += (age_out.argmax(dim=-1) == age_label).sum().item()

        if (batch+1) % 50 == 0:
            train_loss = loss_value / 50
            mask_loss = mask_loss_value / 50
            gender_loss = gender_loss_value / 50
            age_loss = age_loss_value / 50

            mask_acc = mask_match / configs['train']['batch_size'] / 50
            gender_acc = gender_match / configs['train']['batch_size'] / 50
            age_acc = age_match / configs['train']['batch_size'] / 50
            train_acc = (mask_acc + gender_acc + age_acc) / 3

            current_lr = get_lr(optimizer)

            print(
                f"Epoch[{epoch}/{epochs}] ({batch + 1}/{len(dataloader)}) "
                f"| lr {current_lr} \ntrain loss {train_loss:4.2} "
                f"| mask loss {mask_loss:4.2} | gender loss {gender_loss:4.2} "
                f"| age loss {age_loss:4.2}\n"
                f"train acc {train_acc:4.2%} | mask acc {mask_acc:4.2%} "
                f"| gender acc {gender_acc:4.2%} | age acc {age_acc:4.2%}"
            )
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "mask_loss": mask_loss,
                "mask_acc": mask_acc,
                "gender_loss": gender_loss,
                "gender_acc": gender_acc,
                "age_loss": age_loss,
                "age_acc": age_acc,
                'train_rgb': wandb.Image(images[0], caption=f'{targets[0]}')
            })

            loss_value = 0
            mask_loss_value = 0
            gender_loss_value = 0
            age_loss_value = 0

            mask_match = 0
            gender_match = 0
            age_match = 0

    if scheduler is not None:
        scheduler.step()


def validation(
    save_dir: os.PathLike,
    dataloader: DataLoader,
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

    val_mask_losses = []
    val_gender_losses = []
    val_age_losses = []
    val_losses = []

    mask_matches = []
    gender_matches = []
    age_matches = []
    accuracy_values = []

    example_images = []
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            mask_label, gender_label, age_label = targets
            mask_label = mask_label.long().to(device)
            gender_label = gender_label.long().to(device)
            age_label = age_label.long().to(device)

            mask_out, gen_out, age_out = model(images)
            mask_loss = loss_fn(mask_out, mask_label)
            gender_loss = loss_fn(gen_out, gender_label)
            age_loss = loss_fn(age_out, age_label)
            total_loss = mask_loss + gender_loss + age_loss

            val_mask_losses.append(mask_loss.item())
            val_gender_losses.append(gender_loss.item())
            val_age_losses.append(age_loss.item())
            val_losses.append(total_loss.item())

            mask_match = (mask_out.argmax(dim=-1) == mask_label).sum().item()
            gen_match = (gen_out.argmax(dim=-1) == gender_label).sum().item()
            age_match = (age_out.argmax(dim=-1) == age_label).sum().item()
            mask_matches.append(mask_match)
            gender_matches.append(gen_match)
            age_matches.append(age_match)
            accuracy_values.append((mask_match + gen_match + age_match) / 3)

            labels = mask_label * 6 + gender_label * 3 + age_label
            preds = mask_out.argmax(dim=-1) * 6 + gen_out.argmax(dim=-1) * 3 \
                + age_out.argmax(dim=-1)
            val_labels.extend(labels.cpu().numpy())  # labal,pred 저장
            val_preds.extend(preds.cpu().numpy())

            if (batch+1) % 50 == 0:
                idx = random.randint(0, len(mask_out))
                outputs = str(mask_out[idx].cpu().numpy()) + \
                    str(gen_out[idx].cpu().numpy()) + str(age_out[idx].cpu().numpy())
                targets = str(mask_label[idx].cpu().numpy()) + \
                    str(gender_label[idx].cpu().numpy()) + str(age_label[idx].cpu().numpy())
                example_images.append(
                    wandb.Image(
                        images[idx], caption="Pred:{} Truth:{}".format(outputs, targets)
                    )
                )
                wandb.log({"Image": example_images})

    val_loss = np.sum(val_losses) / len(dataloader)
    val_mask_loss = np.sum(val_mask_losses) / len(dataloader)
    val_gender_loss = np.sum(val_gender_losses) / len(dataloader)
    val_age_loss = np.sum(val_age_losses) / len(dataloader)

    val_acc = np.sum(accuracy_values) / len(dataloader.dataset)
    mask_acc = np.sum(mask_matches) / len(dataloader.dataset)
    gender_acc = np.sum(gender_matches) / len(dataloader.dataset)
    age_acc = np.sum(age_matches) / len(dataloader.dataset)

    val_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')

    print(
        f"Epoch[{epoch}]({len(dataloader)})\n"
        f"valid loss {val_loss:4.2} | mask loss {val_mask_loss:4.2} "
        f"| gender loss {val_gender_loss:4.2} | age loss {val_age_loss:4.2}"
        f"\nvalid acc {val_acc:4.2%} | mask acc {mask_acc:4.2%} "
        f"| gender acc {gender_acc:4.2%} | age acc {age_acc:4.2%}"
        f"\nvalid f1 score {val_f1:.5}"
    )
    wandb.log({
        "valid_loss": val_loss,
        "valid_acc": val_acc,
        "valid_mask_loss": val_mask_loss,
        "mask_acc": mask_acc,
        "valid_gender_loss": val_gender_loss,
        "gender_acc": gender_acc,
        "valid_age_loss": val_age_loss,
        "age_acc": age_acc,
        "val_f1_score": val_f1
    })
    torch.save(
        model.state_dict(),
        f'{save_dir}/{epoch}-{val_loss:4.2}-{val_acc:4.2}.pth'
        )
    print(
        f'Saved Model to {save_dir}/{epoch}-{val_loss:4.2}-{val_acc:4.2}.pth'
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
    image_size = configs['data']['image_size']
    train_augmentation = TrainAugmentation(resize=[image_size, image_size])
    train_data = MultiLabelDataset(
        image_dir=configs['data']['train_dir'],
        csv_path=configs['data']['csv_dir'],
        transform=train_augmentation,
        mode='train',
        valid_rate=configs['data']['valid_rate']
    )

    valid_augmentation = TestAugmentation(resize=[image_size, image_size])
    val_data = MultiLabelDataset(
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
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MultiLabelModel().to(device)

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
            configs, train_loader, device,
            model, loss_fn, optimizer, scheduler, e+1
        )
        validation(save_dir, val_loader, device, model, loss_fn, e+1)
        print('\n')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/multi_label_train.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    seed_everything(configs['seed'])
    run_pytorch(configs=configs)