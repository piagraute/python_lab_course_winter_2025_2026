import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2(num_classes: int = 43, is_pretrained: bool = False):
    if is_pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
    else:
        weights = None

    model = mobilenet_v2(weights=weights)

    # set output layer for amount of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_mobilenet_v2()
    print(model)
