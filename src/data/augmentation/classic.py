from torchvision.transforms import Compose
from typing import Literal
from src.data.augmentation.baseline import create_baseline_augmentation


def create_classic_augmentation(model: Literal["cnn", "mobilenetv2"]) -> Compose:
    baseline_aug = create_baseline_augmentation(model)
    ...
    #TODO
