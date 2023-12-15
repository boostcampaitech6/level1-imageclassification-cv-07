import argparse
import random

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.mask_dataset import MaskDatasetV1
from models.mask_model import MaskModelV3
from utils.transform import TestAugmentation


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
    submission: pd.DataFrame
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

            output1, output2, output3 = model(images)
            output1 = output1.argmax(dim=-1)
            output2 = output2.argmax(dim=-1)
            output3 = output3.argmax(dim=-1)
            output = output1 + output2 + output3
            predicts.extend(output.cpu().numpy())
    submission['ans'] = predicts
    submission.to_csv()


def run_pytorch(configs) -> None:
    """추론 파이토치 파이프라인

    :param configs: 학습에 사용할 config들
    :type configs: dict
    """
    pd.read_csv(configs['data']['csv_dir'])
    image_size = configs['data']['image_size']
    test_augmentation = TestAugmentation(resize=[image_size, image_size])
    test_data = MaskDatasetV1(
        image_dir=configs['data']['test_dir'],
        transform=test_augmentation,
        mode='test',
    )

    test_loader = DataLoader(
        test_data,
        shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskModelV3().to(device)
    model.load_state_dict(torch.load(configs['ckpt_path']))
    predict(test_loader, device, model)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="test.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    seed_everything(configs['seed'])
    run_pytorch(configs=configs)
