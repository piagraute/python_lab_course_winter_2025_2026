from typing import Literal 
from src.models.cnn import CNN
from src.models.mobilenet import build_mobilenet_v2
from logging import Logger

def load_model(model_str: Literal["cnn", "mobilenetv2"], logger:Logger):
    if model_str == "cnn":
        return CNN(logger)
    if model_str == "mobilenetv2":
        return build_mobilenet_v2()
    raise ValueError(f"Unknown model type: {model_str}")