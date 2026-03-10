from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from typing import Literal


def create_baseline_augmentation(model: Literal["cnn", "mobilenetv2"]) -> Compose:
    if model == "cnn":
        return Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
        ])
    elif model == "mobilenetv2":
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

