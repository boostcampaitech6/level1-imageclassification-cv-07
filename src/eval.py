import argparse
import random
import os

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.datasets import TestDataset
from models.mask_model import SingleLabelModel


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def predict(
    dataloader: DataLoader,
    device: str,
    model: nn.Module,
    submission: pd.DataFrame,
    save_path: os.PathLike
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
    predicts = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)

            output = model(images)
            output = output.argmax(dim=-1)
            predicts.extend(output.cpu().numpy())
    submission['ans'] = predicts
    submission.to_csv(save_path, index=False)


def run_pytorch(configs) -> None:
    """추론 파이토치 파이프라인

    :param configs: 학습에 사용할 config들
    :type configs: dict
    """
    submission = pd.read_csv(configs['data']['csv_dir'])
    width, height = map(int, configs['data']['image_size'].split(','))
    image_paths = [os.path.join(configs['data']['test_dir'], img_id) for img_id in submission.ImageID]
    test_data = TestDataset(
        img_paths=image_paths,
        width=width,
        height=height
    )

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
        version = 'submission_v' + str(i) + '.csv'
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
