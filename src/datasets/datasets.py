import os
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:

            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 58:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
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
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        valid_rate=0.2,
    ) -> None:
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.val_ratio = valid_rate
        self.indices = defaultdict(list)
        self.transform = None
        self.mean = None
        self.std = None

        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.image_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.image_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)

                if (
                    _file_name not in self._file_names
                ):
                    continue

                img_path = os.path.join(
                    self.image_dir, profile, file_name
                )
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics..."
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class MultiLabelMaskBaseDataset(Dataset):
    num_classes = 3 + 2 + 3

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        valid_rate=0.2,
    ) -> None:
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.val_ratio = valid_rate
        self.indices = defaultdict(list)
        self.transform = None
        self.mean = None
        self.std = None

        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.image_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.image_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)

                if (
                    _file_name not in self._file_names
                ):
                    continue

                img_path = os.path.join(
                    self.image_dir, profile, file_name
                )
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics..."
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class MaskSplitByProfileDataset(Dataset):
    num_classes = 3 * 2 * 3

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    total_labels = []
    train_labels = []
    val_labels = []

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        valid_rate=0.2,
    ) -> None:
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.val_ratio = valid_rate
        self.indices = defaultdict(list)
        self.transform = None
        self.mean = None
        self.std = None

        self.setup()
        self.calc_statistics()

    def split_profile(self, val_ratio):
        df = pd.read_csv(self.csv_path)
        df.drop(['id', 'race'], axis=1, inplace=True)

        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        df['age'] = pd.cut(df['age'], bins=[-float('inf'), 30, 60, float('inf')], labels=[0, 1, 2], right=False).astype(int)

        df['gender_age'] = df['gender'].astype(int) * 3 + df['age'].astype(int)

        train, val = train_test_split(
            df, test_size=val_ratio, random_state=42, stratify=df["gender_age"]
        )
        train_indices = set(list(train.index))
        val_indices = set(list(val.index))

        return {"train": train_indices, "val": val_indices}

    def setup(self):
        profiles = os.listdir(self.image_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]

        split_profiles = self.split_profile(self.val_ratio)
        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.image_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):
                        continue

                    img_path = os.path.join(
                        self.image_dir, profile, file_name
                    )
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.total_labels.append(
                        self.encode_multi_class(mask_label, gender_label, age_label)
                    )

                    self.indices[phase].append(cnt)
                    cnt += 1

        for phase, indices in self.indices.items():
            if phase == "train":
                for t in indices:
                    self.train_labels.append(self.total_labels[t])
            else:
                for v in indices:
                    self.val_labels.append(self.total_labels[v])

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics..."
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        target = self.encode_multi_class(mask_label, gender_label, age_label)

        image = np.array(image)
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)['image']

        return image, target

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class MultiLabelMaskSplitByProfileDataset(Dataset):
    total_labels = []
    train_labels = []
    val_labels = []
    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        image_dir: os.PathLike,
        csv_path: os.PathLike,
        valid_rate=0.2,
    ) -> None:
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.val_ratio = valid_rate
        self.indices = defaultdict(list)
        self.transform = None
        self.mean = None
        self.std = None

        self.setup()
        self.calc_statistics()

    def split_profile(self, val_ratio):
        df = pd.read_csv(self.csv_path)
        df.drop(['id', 'race'], axis=1, inplace=True)

        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        df['age'] = pd.cut(df['age'], bins=[-float('inf'), 30, 60, float('inf')],
                        labels=[0, 1, 2], right=False).astype(int)

        df['gender_age'] = df['gender'].astype(int) * 3 + df['age'].astype(int)

        train, val = train_test_split(
            df, test_size=val_ratio, random_state=42, stratify=df["gender_age"]
        )
        train_indices = set(list(train.index))
        val_indices = set(list(val.index))

        return {"train": train_indices, "val": val_indices}

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics..."
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def setup(self):
        profiles = os.listdir(self.image_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]

        split_profiles = self.split_profile(self.val_ratio)
        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.image_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):
                        continue

                    img_path = os.path.join(
                        self.image_dir, profile, file_name
                    )
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    self.total_labels.append(
                        [mask_label, gender_label, age_label]
                    )

                    self.indices[phase].append(cnt)
                    cnt += 1

        for phase, indices in self.indices.items():
            if phase == "train":
                for t in indices:
                    self.train_labels.append(self.total_labels[t])
            else:
                for v in indices:
                    self.val_labels.append(self.total_labels[v])

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        target = [mask_label, gender_label, age_label]

        image = np.array(image)
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)['image']

        return image, target

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path).convert('RGB')

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class TestDataset(Dataset):
    def __init__(
        self, img_paths, width, height, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = A.Compose(
            [
                A.Resize(width, height),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return len(self.img_paths)


class TTADataset(Dataset):
    def __init__(
        self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.transform = A.Compose(
            [
                A.CenterCrop((380, 380)),
                A.Resize(resize, Image.BILINEAR),
                ToTensorV2(),
                A.Normalize(mean=mean, std=std),
            ]
        )
        print("TTADataset is Used")

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
