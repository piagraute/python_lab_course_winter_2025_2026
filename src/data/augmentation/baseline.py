from torchvision.transforms import Compose
from typing import Literal


def create_baseline_augmentation(model: Literal["cnn", "mobilenetv2"]) -> Compose:
    ...
    #TODO
