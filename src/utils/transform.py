from typing import Tuple, List

import albumentations as A
from albumentations.pytorch import ToTensorV2


# A.Resize(512, 512),
# A.CenterCrop(380, 380),
# A.HorizontalFlip(p=0.5),
# A.RandomRotate90(p=0.5),
#     A.VerticalFlip(p=0.5)
# A.OneOf([
#     A.MotionBlur(p=0.5),
#     A.OpticalDistortion(p=0.5),
#     A.GaussNoise(p=0.5)
# ], p=1),
# A.ColorJitter(p=0.2),
# A.RGBShift(p=0.2),
# A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
# ToTensorV2()


class TrainAugmentation:
    """
    기본적인 Augmentation을 담당하는 클래스
    Attributes:
        transform (Compose): 이미지를 변환을 위한 torchvision.transforms.Compose 객체
    """

    def __init__(
        self, resize: List = [380, 380], mean: Tuple = [0.548, 0.504, 0.479],
        std: Tuple = [0.237, 0.247, 0.246], **args
    ):
        """
        Args:
            resize (tuple): 이미지의 리사이즈 대상 크지
            mean (tuple): Normalize 변환을 위한 평균 값
            std (tuple): Normalize 변환을 위한 표준 값
        """
        width, height = resize
        self.transform = A.Compose(
            [
                A.Resize(width, height),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용
        Args:
            Image (PIL.Image): Augumentation을 적용할 이미지
        Returns:
            Tensor: Argumentation이 적용된 이미지
        """
        return self.transform(image=image)


class TestAugmentation:
    """
    기본적인 Augmentation을 담당하는 클래스
    Attributes:
        transform (Compose): 이미지를 변환을 위한 torchvision.transforms.Compose 객체
    """

    def __init__(
        self, resize: List = (380, 380), mean: Tuple = (0.548, 0.504, 0.479),
        std: Tuple = (0.237, 0.247, 0.246), **args
    ):
        """
        Args:
            resize (tuple): 이미지의 리사이즈 대상 크지
            mean (tuple): Normalize 변환을 위한 평균 값
            std (tuple): Normalize 변환을 위한 표준 값
        """
        width, height = resize
        self.transform = A.Compose(
            [
                A.Resize(width, height),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용
        Args:
            Image (PIL.Image): Augumentation을 적용할 이미지
        Returns:
            Tensor: Argumentation이 적용된 이미지
        """
        return self.transform(image=image)
