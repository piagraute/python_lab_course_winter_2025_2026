from torchvision.transforms import Compose
from typing import Literal
from src.data.augmentation.baseline import create_baseline_augmentation
from torchvision.transforms import ColorJitter, RandomRotation, RandomAffine


def create_classic_augmentation(model: Literal["cnn", "mobilenetv2"]) -> Compose:
    baseline_aug = create_baseline_augmentation(model)
    return Compose([
        baseline_aug,
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomRotation(degrees=15),
        RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])


