import argparse
import os
from typing import Dict

from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.mask_datasets import TestDataset
from models.mask_model import SingleLabelModel
from utils.utils import seed_everything


def predict(
    dataloader: DataLoader,
    device: str,
    model: nn.Module,
    submission: pd.DataFrame,
    save_path: os.PathLike
) -> None:
    """
    데이터셋으로 테스트

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param submission: submission csv 데이터프레임
    :type submission: pd.DataFrame
    :param save_path: csv 저장할 경로
    :type save_path: os.PathLike
    """
    model.eval()
    predicts = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)

            output = model(images)
            output = output.argmax(dim=-1)
            predicts.extend(output.cpu().numpy())
    submission['ans'] = predicts
    submission.to_csv(save_path, index=False)


def run_pytorch(configs: Dict) -> None:
    """
    추론 파이토치 파이프라인

    :param configs: 학습에 사용할 config
    :type configs: dict
    """
    test_dir = configs['data']['test_dir']
    submission = pd.read_csv(configs['data']['csv_dir'])
    image_paths = [
        os.path.join(test_dir, img_id) for img_id in submission.ImageID
    ]
    width, height = map(int, configs['data']['image_size'].split(','))
    test_data = TestDataset(
        image_paths=image_paths,
    )
    if configs['train']['imagenet']:
        mean = [0.548, 0.504, 0.479]
        std = [0.237, 0.247, 0.246]
    else:
        mean = [0.561, 0.524, 0.501]
        std = [0.233, 0.243, 0.246]
    test_transforms = A.Compose([
        A.Resize(width, height),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    test_data.set_transform(test_transforms)

    test_loader = DataLoader(
        test_data,
        shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SingleLabelModel().to(device)
    model.load_state_dict(torch.load(configs['ckpt_path']))

    if not os.path.exists(os.path.join('results', model.name)):
        os.makedirs(os.path.join('results', model.name))

    i = 0
    while True:
        version = str(configs['model']) + str(configs['train']['loss']) \
            + str(configs['train']['optim']) + str(i) + '.csv'
        if os.path.exists(os.path.join('results', model.name, version)):
            i += 1
            continue
        else:
            save_path = os.path.join('results', model.name, version)
            break
    predict(test_loader, device, model, submission, save_path)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/test.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    seed_everything(configs['seed'])
    run_pytorch(configs=configs)
