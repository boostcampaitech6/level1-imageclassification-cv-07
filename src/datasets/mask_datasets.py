import os
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Dict

from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
import albumentations as A


class MaskLabels(int, Enum):
    """
    MaskLabels 클래스 정의
    정상 착용 = 0
    오착용 = 1
    미착용 = 2
    """
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """
    GenderLabels 클래스 정의
    남자 = 0
    여자 = 1
    """
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> Enum:
        """
        get genderclass method

        :param value: 성별 레이블의 문자열
        :type value: str
        :return: 성별 레이블의 class
        :rtype: Enum
        """
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either male or female, {value}"
            )


class AgeLabels(int, Enum):
    """
    AgeLabels 클래스 정의
    age < 30 = 0
    30 <= age < 60 = 1
    60 < age = 2
    """
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> Enum:
        """
        get ageclass method

        :param value: 나이 레이블의 문자열
        :type value: str
        :return: 나이 레이블의 class
        :rtype: Enum
        """
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class SingleLabelDataset(Dataset):
    """
    SingleLabel Dataset 클래스 정의
    이미지와 정답 불러와서 각각 Tensor와 encoding 후 값 반환
    """
    num_classes = 3 * 2 * 3

    file_mask_label = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self, root_folder: os.PathLike, valid_rate: float = 0.2,
    ) -> None:
        """
        define SingleLabelDataset

        :param root_folder: 데이터 폴더 상단 ex)'./data/train/images'
        :type root_folder: os.PathLike
        :param valid_rate: train과 valid 데이터 나누는 기준 값
        :type valid_rate: float
        """
        self.root_folder = root_folder
        self.valid_rate = valid_rate
        self.transform = None
        self.mean = None
        self.std = None

        self.setup()
        self.calc_statistics()

    def setup(self) -> None:
        """
        이미지 경로 설정 및 마스크, 성별, 나이 라벨 설정
        """
        profiles = os.listdir(self.root_folder)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.root_folder, profile)
            for file_name in os.listdir(img_folder):
                _file_name, _ = os.path.splitext(file_name)

                if _file_name not in self.file_mask_label:
                    continue

                img_path = os.path.join(self.root_folder, profile, file_name)
                mask_label = self.file_mask_label[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self) -> None:
        """
        이미지의 평균과 표준편차 구하기
        """
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics...")
            sums = []
            squared = []
            for image_path in tqdm(self.image_paths[:3000]):
                image = np.array(Image.open(image_path)).astype(np.uint8)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform: A.Compose) -> None:
        """
        transform setter method
        :param transform: albumentations transform compose한 것
        :type transform: A.Compose
        """
        self.transform = transform

    def __len__(self) -> int:
        """
        len special method
        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        getitem special method
        :param index: 데이터 인덱스
        :type index: int
        :return: 데이터 하나의 image와 label이 Tuple로 묶임
        :rtype: Tuple[Tensor, Tensor]
        """
        assert self.transform is not None

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        target = self.encode_multi_class(mask_label, gender_label, age_label)

        image = np.array(image)
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image)['image']

        return image, target

    def get_mask_label(self, index: int) -> MaskLabels:
        """
        get mask label method
        :param index: 데이터 인덱스
        :type index: int
        :return: MaskLabels 인덱스 해당 값 반환
        :rtype: MaskLabels
        """
        return self.mask_labels[index]

    def get_gender_label(self, index: int) -> GenderLabels:
        """
        get gender label method
        :param index: 데이터 인덱스
        :type index: int
        :return: GenderLabels 인덱스 해당 값 반환
        :rtype: GenderLabels
        """
        return self.gender_labels[index]

    def get_age_label(self, index: int) -> AgeLabels:
        """
        get age label method
        :param index: 데이터 인덱스
        :type index: int
        :return: AgeLabels 인덱스 해당 값 반환
        :rtype: AgeLabels
        """
        return self.age_labels[index]

    def read_image(self, index: int) -> Image.Image:
        """
        get image method
        :param index: 데이터 인덱스
        :type index: int
        :return: 사진 데이터 인덱스 해당 값 반환
        :rtype: Image.Image
        """
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def encode_multi_class(
        mask_label: MaskLabels, gender_label: GenderLabels,
        age_label: AgeLabels
    ) -> int:
        """
        get encode label method
        :param mask_label: 마스크 라벨
        :type mask_label: MaskLabels
        :param gender_label: 성별 라벨
        :type gender_label: GenderLabels
        :param age_label: 나이 라벨
        :type age_label: AgeLabels
        :return: encoding한 라벨 반환
        :rtype: int
        """
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label: int,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        """
        get decode label method
        :param multi_class_label: encoding한 라벨
        :type multi_class_label: int
        :return: decode한 라벨 Tuple로 묶어 반환
        :rtype: Tuple[MaskLabels, GenderLabels, AgeLabels]
        """
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    def split_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        get split dataset method
        :return: 데이터셋 나눠서 Tuple로 묶기
        :rtype: Tuple[Dataset, Dataset]
        """
        n_val = int(len(self) * self.valid_rate)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MultiLabelDataset(Dataset):
    """
    MultiLabel Dataset 클래스 정의
    이미지와 정답 불러와서 각각 Tensor와 encoding 후 값 반환
    """
    num_classes = 3 + 2 + 3

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    file_mask_label = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self, root_folder: os.PathLike, valid_rate: float = 0.2
    ) -> None:
        """
        define MultiLabelDataset

        :param root_folder: 데이터 폴더 상단 ex)'./data/train/images'
        :type root_folder: os.PathLike
        :param valid_rate: train과 valid 데이터 나누는 기준 값
        :type valid_rate: float
        """
        self.root_folder = root_folder
        self.valid_rate = valid_rate
        self.indices = defaultdict(list)
        self.transform = None
        self.mean = None
        self.std = None

        self.setup()
        self.calc_statistics()

    def setup(self) -> None:
        """
        이미지 경로 설정 및 마스크, 성별, 나이 라벨 설정
        """
        profiles = os.listdir(self.root_folder)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.root_folder, profile)
            for file_name in os.listdir(img_folder):
                _file_name, _ = os.path.splitext(file_name)

                if _file_name not in self.file_mask_label:
                    continue

                img_path = os.path.join(self.root_folder, profile, file_name)
                mask_label = self.file_mask_label[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self) -> None:
        """
        이미지의 평균과 표준편차 구하기
        """
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics...")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform: A.Compose) -> None:
        """
        transform setter method
        :param transform: albumentations transform compose한 것
        :type transform: A.Compose
        """
        self.transform = transform

    def __len__(self) -> int:
        """
        len special method
        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        getitem special method
        :param index: 데이터 인덱스
        :type index: int
        :return: 데이터 하나의 image와 label이 Tuple로 묶임
        :rtype: Tuple[Tensor, Tensor]
        """
        assert self.transform is not None

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        target = [mask_label, gender_label, age_label]

        image = np.array(image)
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image)['image']

        return image, target

    def get_mask_label(self, index: int) -> int:
        """
        get mask label method
        :param index: 데이터 인덱스
        :type index: int
        :return: MaskLabels 인덱스 해당 값 반환
        :rtype: int
        """
        return int(self.mask_labels[index])

    def get_gender_label(self, index: int) -> int:
        """
        get gender label method
        :param index: 데이터 인덱스
        :type index: int
        :return: GenderLabels 인덱스 해당 값 반환
        :rtype: int
        """
        return int(self.gender_labels[index])

    def get_age_label(self, index: int) -> int:
        """
        get age label method
        :param index: 데이터 인덱스
        :type index: int
        :return: AgeLabels 인덱스 해당 값 반환
        :rtype: int
        """
        return int(self.age_labels[index])

    def read_image(self, index: int) -> Image.Image:
        """
        get image method
        :param index: 데이터 인덱스
        :type index: int
        :return: 사진 데이터 인덱스 해당 값 반환
        :rtype: Image.Image
        """
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def encode_multi_class(
        mask_label: MaskLabels, gender_label: GenderLabels,
        age_label: AgeLabels
    ) -> int:
        """
        get encode label method
        :param mask_label: 마스크 라벨
        :type mask_label: MaskLabels
        :param gender_label: 성별 라벨
        :type gender_label: GenderLabels
        :param age_label: 나이 라벨
        :type age_label: AgeLabels
        :return: encoding한 라벨 반환
        :rtype: int
        """
        return mask_label * 6 + gender_label * 3 + age_label

    def split_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        get split dataset method
        :return: 데이터셋 나눠서 Tuple로 묶기
        :rtype: Tuple[Dataset, Dataset]
        """
        n_val = int(len(self) * self.valid_rate)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(SingleLabelDataset):
    """
    MaskSplitByProfile Dataset 클래스 정의
    이미지와 정답 불러와서 각각 Tensor와 encoding 후 값 반환
    """
    def __init__(
        self, root_folder: os.PathLike,
        valid_rate: float = 0.2, csv_path: os.PathLike = None
    ) -> None:
        """
        define MaskSplitByProfileDataset

        :param root_folder: 데이터 폴더 상단 ex)'./data/train/images'
        :type root_folder: os.PathLike
        :param valid_rate: train과 valid 데이터 나누는 기준 값
        :type valid_rate: float
        :param csv_path: csv 파일 경로
        :type csv_path: os.PathLike
        """
        self.csv_path = csv_path
        self.indices = defaultdict(list)
        super().__init__(root_folder, valid_rate)

    def split_profile(self) -> Dict:
        """
        사람 기준으로 train, val 나누기
        :return: 데이터셋 경로 나눠서 Dict로 묶기
        :rtype: Dict
        """
        df = pd.read_csv(self.csv_path)
        df.drop(['id', 'race'], axis=1, inplace=True)

        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        df['age'] = pd.cut(
            df['age'], bins=[-float('inf'), 30, 60, float('inf')],
            labels=[0, 1, 2], right=False
        ).astype(int)

        df['gender_age'] = df['gender'].astype(int) * 3 + df['age'].astype(int)

        train, val = train_test_split(
            df, test_size=self.valid_rate,
            random_state=42, stratify=df["gender_age"]
        )
        train_indices = set(list(train.index))
        val_indices = set(list(val.index))

        return {"train": train_indices, "val": val_indices}

    def setup(self) -> None:
        """
        사람 기준으로 가져온 데이터 split하여 이미지 경로 설정 및 마스크, 성별, 나이 라벨 설정
        """
        profiles = os.listdir(self.root_folder)
        profiles = [
            profile for profile in profiles if not profile.startswith(".")
        ]

        split_profiles = self.split_profile(self.valid_rate)
        cnt = 0
        for phase, indices in split_profiles.items():
            for idx in indices:
                profile = profiles[idx]
                img_folder = os.path.join(self.root_folder, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, _ = os.path.splitext(file_name)
                    if _file_name not in self.file_mask_label:
                        continue

                    img_path = os.path.join(
                        self.root_folder, profile, file_name
                    )
                    mask_label = self.file_mask_label[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset, Subset]:
        """
        get split dataset method
        :return: 데이터셋 나눠서 List로 묶기
        :rtype: List[Subset, Subset]
        """
        return [
            Subset(self, indices) for phase, indices in self.indices.items()
        ]


class MultiLabelMaskSplitByProfileDataset(MultiLabelDataset):
    """
    MultiLabelMaskSplitByProfileDataset Dataset 클래스 정의
    이미지와 정답 불러와서 각각 Tensor와 encoding 후 값 반환
    """
    def __init__(
        self, root_folder: os.PathLike,
        valid_rate: float = 0.2, csv_path: os.PathLike = None
    ) -> None:
        """
        define MultiLabelMaskSplitByProfileDataset

        :param root_folder: 데이터 폴더 상단 ex)'./data/train/images'
        :type root_folder: os.PathLike
        :param valid_rate: train과 valid 데이터 나누는 기준 값
        :type valid_rate: float
        :param csv_path: csv 파일 경로
        :type csv_path: os.PathLike
        """
        self.csv_path = csv_path
        self.indices = defaultdict(list)
        super().__init__(root_folder, valid_rate)

    def split_profile(self) -> Dict:
        """
        사람 기준으로 train, val 나누기
        :return: 데이터셋 경로 나눠서 Dict로 묶기
        :rtype: Dict
        """
        df = pd.read_csv(self.csv_path)
        df.drop(['id', 'race'], axis=1, inplace=True)

        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        df['age'] = pd.cut(
            df['age'], bins=[-float('inf'), 30, 60, float('inf')],
            labels=[0, 1, 2], right=False
        ).astype(int)

        df['gender_age'] = df['gender'].astype(int) * 3 + df['age'].astype(int)

        train, val = train_test_split(
            df, test_size=self.valid_rate,
            random_state=42, stratify=df["gender_age"]
        )
        train_indices = set(list(train.index))
        val_indices = set(list(val.index))

        return {"train": train_indices, "val": val_indices}

    def setup(self) -> None:
        """
        사람 기준으로 가져온 데이터 split하여 이미지 경로 설정 및 마스크, 성별, 나이 라벨 설정
        """
        profiles = os.listdir(self.root_folder)
        profiles = [
            profile for profile in profiles if not profile.startswith(".")
        ]

        split_profiles = self.split_profile(self.valid_rate)
        cnt = 0
        for phase, indices in split_profiles.items():
            for idx in indices:
                profile = profiles[idx]
                img_folder = os.path.join(self.root_folder, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, _ = os.path.splitext(file_name)
                    if _file_name not in self.file_mask_label:
                        continue

                    img_path = os.path.join(
                        self.root_folder, profile, file_name
                    )
                    mask_label = self.file_mask_label[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset, Subset]:
        """
        get split dataset method
        :return: 데이터셋 나눠서 List로 묶기
        :rtype: List[Subset, Subset]
        """
        return [
            Subset(self, indices) for phase, indices in self.indices.items()
        ]


class TestDataset(Dataset):
    """
    Test Dataset 클래스 정의
    이미지와 불러와서 Tensor로 값 반환
    """
    def __init__(self, image_paths: List) -> None:
        self.image_paths = image_paths
        self.transform = None

    def set_transform(self, transform: A.Compose) -> None:
        """
        transform setter method
        :param transform: albumentations transform compose한 것
        :type transform: A.Compose
        """
        self.transform = transform

    def __len__(self) -> int:
        """
        len special method
        :return: 데이터셋 길이
        :rtype: int
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        """
        getitem special method
        :param index: 데이터 인덱스
        :type index: int
        :return: 데이터 image 인덱스에 맞게 반환
        :rtype: Tensor
        """
        image = Image.open(self.image_paths[index])
        image = np.array(image)
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)['image']
        return image


class TTADataset(Dataset):
    """
    TTA Dataset 클래스 정의
    이미지와 불러와서 Tensor로 값 반환
    """
    pass
