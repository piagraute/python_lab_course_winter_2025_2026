from torchvision.transforms import AutoAugment, AutoAugmentPolicy, Compose
from typing import Literal
from src.data.augmentation.baseline import create_baseline_augmentation


def create_advanced_augmentation(model: Literal["cnn", "mobilenetv2"]) -> Compose:
    baseline_aug = create_baseline_augmentation(model)
    return Compose([
        baseline_aug,
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
    ])
    
