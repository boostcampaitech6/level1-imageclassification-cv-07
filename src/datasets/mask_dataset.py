import os
import csv
import random
from typing import Sequence, Callable, Tuple

import numpy as np
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset


classes = [
    'incorrect_mask', 'mask1', 'mask2', 'mask3',
    'mask4', 'mask5', 'normal'
    ]


def one_hot_mask(label):
    if 'incorrect_mask' in label:
        label = [1, 0, 0]
    elif 'mask' in label:
        label = [0, 1, 0]
    elif 'normal' in label:
        label = [0, 0, 1]
    else:
        raise ValueError('This image has mask info?')
    return label


def one_hot_gender(label):
    label = label.lower()
    if label == 'male':
        label = [1, 0]
    elif label == 'female':
        label = [0, 1]
    else:
        raise ValueError('This image has gender?')
    return label


def one_hot_age(label):
    label = int(label)
    if label < 30:
        label = [1, 0, 0]
    elif label < 60:
        label = [0, 1, 0]
    else:
        label = [0, 0, 1]
    return label


def MaskLabels(label):
    if 'incorrect_mask' in label:
        label = 0
    elif 'mask' in label:
        label = 1
    elif 'normal' in label:
        label = 2
    else:
        raise ValueError('This image has mask info?')
    return label


def GenderLabels(label):
    label = label.lower()
    if label == "male":
        return 0
    elif label == "female":
        return 1
    else:
        raise ValueError(
            f"Gender value should be either 'male' or 'female', {label}"
        )


def AgeLabels(label):
    label = int(label)
    if label < 30:
        return 0
    elif label < 60:
        return 1
    elif label >= 60:
        return 2
    else:
        raise ValueError(f"Age value should be numeric, {label}")


class TestDataset(Dataset):
    def __init__(
        self,
        image_paths: os.PathLike,
        transform: Sequence[Callable] = None,
        mode: str = 'train',
    ) -> None:
        """데이터 정보를 불러와 정답(label)과 각각 데이터의 이름(image_id)를 저장

        :param dir: 데이터셋 경로
        :type dir: os.PathLike
        :param image_ids: 데이터셋의 정보가 담겨있는 csv 파일 경로
        :type image_ids: os.PathLike
        :param transforms: 데이터셋을 정규화하거나 텐서로 변환,
        augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transforms: Sequence[Callable]
        """
        super().__init__()

        self.image_paths = image_paths
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        """데이터셋의 길이를 반환

        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """데이터의 인덱스를 주면 이미지와 정답을 같이 반환하는 함수

        :param index: 이미지 인덱스
        :type index: int
        :return: 이미지 한장과 정답 값들
        :rtype: Tuple[Tensor]
        """
        image = Image.open(self.image_paths[index])

        if self.transform:
            image = self.transform(image)
        return image


class MaskDatasetV1(Dataset):
    """데이터셋 사용자 정의 클래스를 정의합니다.
    albumentations 용도이기 때문에 torchvision transform 추가 작성 필요 -> TODO 1
    """
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Sequence[Callable] = None,
        mode: str = 'train',
        valid_rate: float = 0.2,
    ) -> None:
        """데이터 정보를 불러와 정답(label)과 각각 데이터의 이름(image_id)를 저장

        :param dir: 데이터셋 경로
        :type dir: os.PathLike
        :param image_ids: 데이터셋의 정보가 담겨있는 csv 파일 경로
        :type image_ids: os.PathLike
        :param transforms: 데이터셋을 정규화하거나 텐서로 변환,
        augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transforms: Sequence[Callable]
        """
        super().__init__()

        self.image_dir = image_dir
        self.csv_path = csv_path
        self.transform = transform
        self.mode = mode
        self.valid_rate = valid_rate

        self.labels = {}
        self.image_path = []
        self.label_list = []

        self.setup()

    def setup(self):
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[row[0]] = list(row[1:])

        self.image_ids = list(self.labels.keys())

        for image_id in self.image_ids:
            path = self.labels[image_id][-1]
            img_folder = os.path.join(self.image_dir, path)
            for file_name in os.listdir(img_folder):
                _file_name, _ = os.path.splitext(file_name)
                if (_file_name not in classes):
                    continue
                self.image_path.append(
                    os.path.join(self.image_dir, path, file_name)
                )
                self.label_list.append([
                    np.array(MaskLabels(file_name)),
                    np.array(GenderLabels(self.labels[image_id][0])),
                    np.array(AgeLabels(self.labels[image_id][2]))
                ])

        data = list(zip(self.image_path, self.label_list))
        random.shuffle(data)
        self.shuffle_image, self.shuffle_label = zip(*data)
        valid_num = int((1 - float(self.valid_rate)) * len(self.shuffle_image))
        if self.mode == 'train':
            self.shuffle_image = self.shuffle_image[:valid_num]
            self.shuffle_label = self.shuffle_label[:valid_num]

        elif self.mode == 'valid':
            self.shuffle_image = self.shuffle_image[valid_num:]
            self.shuffle_label = self.shuffle_label[valid_num:]

    def __len__(self) -> int:
        """데이터셋의 길이를 반환

        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.shuffle_image)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """데이터의 인덱스를 주면 이미지와 정답을 같이 반환하는 함수

        :param index: 이미지 인덱스
        :type index: int
        :return: 이미지 한장과 정답 값들
        :rtype: Tuple[Tensor]
        """
        path = self.shuffle_image[index]
        target = self.shuffle_label[index]

        image = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
        # image /= 255 -> A.Normalize 적용 안하면 적용해줘야함

        if self.transform:
            image = self.transform(image=image)
            image = image['image']
        return image, target


class MaskDatasetV2(Dataset):
    """데이터셋 사용자 정의 클래스를 정의합니다.
    albumentations 용도이기 때문에 torchvision transform 추가 작성 필요 -> TODO 1
    """
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Sequence[Callable],
        mode: str,
        valid_rate=0.2,
    ) -> None:
        """데이터 정보를 불러와 정답(label)과 각각 데이터의 이름(image_id)를 저장

        :param dir: 데이터셋 경로
        :type dir: os.PathLike
        :param image_ids: 데이터셋의 정보가 담겨있는 csv 파일 경로
        :type image_ids: os.PathLike
        :param transforms: 데이터셋을 정규화하거나 텐서로 변환,
        augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transforms: Sequence[Callable]
        """
        super().__init__()

        self.image_dir = image_dir
        self.csv_path = csv_path
        self.transform = transform
        self.mode = mode
        self.valid_rate = valid_rate

        self.labels = {}
        self.image_path = []
        self.label_list = []
        self.setup()

    def setup(self):
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[row[0]] = list(row[1:])

        self.image_ids = list(self.labels.keys())

        for image_id in self.image_ids:
            path = self.labels[image_id][-1]
            img_folder = os.path.join(self.image_dir, path)
            for file_name in os.listdir(img_folder):
                _file_name, _ = os.path.splitext(file_name)
                if (_file_name not in classes):
                    continue
                self.image_path.append(
                    os.path.join(self.image_dir, path, file_name)
                )
                self.label_list.append(
                    MaskLabels(file_name) * 6 +
                    GenderLabels(self.labels[image_id][0]) * 3 +
                    AgeLabels(self.labels[image_id][2])
                )

        data = list(zip(self.image_path, self.label_list))
        random.shuffle(data)
        self.shuffle_image, self.shuffle_label = zip(*data)
        valid_num = int((1 - float(self.valid_rate)) * len(self.shuffle_image))
        if self.mode == 'train':
            self.shuffle_image = self.shuffle_image[:valid_num]
            self.shuffle_label = self.shuffle_label[:valid_num]

        elif self.mode == 'valid':
            self.shuffle_image = self.shuffle_image[valid_num:]
            self.shuffle_label = self.shuffle_label[valid_num:]

    def __len__(self) -> int:
        """데이터셋의 길이를 반환

        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.shuffle_image)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """데이터의 인덱스를 주면 이미지와 정답을 같이 반환하는 함수

        :param index: 이미지 인덱스
        :type index: int
        :return: 이미지 한장과 정답 값들
        :rtype: Tuple[Tensor]
        """
        path = self.shuffle_image[index]
        target = self.shuffle_label[index]
        targets = [0]*18
        targets[target] = 1
        targets = np.array(targets)

        image = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
        # image /= 255

        if self.transform is not None:
            image = self.transform(image=image)
            image = image['image']
        return image, targets


class MaskDatasetV3(Dataset):
    """데이터셋 사용자 정의 클래스를 정의합니다.
    albumentations 용도이기 때문에 torchvision transform 추가 작성 필요 -> TODO 1
    """
    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        transform: Sequence[Callable],
        mode: str,
        valid_rate=0.2,
    ) -> None:
        """데이터 정보를 불러와 정답(label)과 각각 데이터의 이름(image_id)를 저장

        :param dir: 데이터셋 경로
        :type dir: os.PathLike
        :param image_ids: 데이터셋의 정보가 담겨있는 csv 파일 경로
        :type image_ids: os.PathLike
        :param transforms: 데이터셋을 정규화하거나 텐서로 변환,
        augmentation등의 전처리하기 위해 사용할 여러 함수들의 sequence
        :type transforms: Sequence[Callable]
        """
        super().__init__()

        self.image_dir = image_dir
        self.csv_path = csv_path
        self.transform = transform
        self.mode = mode
        self.valid_rate = valid_rate

        self.labels = {}
        self.image_path = []
        self.label_list = []
        self.setup()

    def setup(self):
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[row[0]] = list(row[1:])

        self.image_ids = list(self.labels.keys())

        for image_id in self.image_ids:
            path = self.labels[image_id][-1]
            img_folder = os.path.join(self.image_dir, path)
            for file_name in os.listdir(img_folder):
                _file_name, _ = os.path.splitext(file_name)
                if (_file_name not in classes):
                    continue
                self.image_path.append(
                    os.path.join(self.image_dir, path, file_name)
                )
                self.label_list.append(
                    np.array(
                        one_hot_mask(file_name) +
                        one_hot_gender(self.labels[image_id][0]) +
                        one_hot_age(self.labels[image_id][2]))
                )
        data = list(zip(self.image_path, self.label_list))
        random.shuffle(data)
        self.shuffle_image, self.shuffle_label = zip(*data)
        valid_num = int((1 - float(self.valid_rate)) * len(self.shuffle_image))
        if self.mode == 'train':
            self.shuffle_image = self.shuffle_image[:valid_num]
            self.shuffle_label = self.shuffle_label[:valid_num]

        elif self.mode == 'valid':
            self.shuffle_image = self.shuffle_image[valid_num:]
            self.shuffle_label = self.shuffle_label[valid_num:]

    def __len__(self) -> int:
        """데이터셋의 길이를 반환

        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.shuffle_image)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        """데이터의 인덱스를 주면 이미지와 정답을 같이 반환하는 함수

        :param index: 이미지 인덱스
        :type index: int
        :return: 이미지 한장과 정답 값들
        :rtype: Tuple[Tensor]
        """
        path = self.shuffle_image[index]
        target = self.shuffle_label[index]

        image = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
        # image /= 255

        if self.transform is not None:
            image = self.transform(image=image)
            image = image['image']
        return image, target
