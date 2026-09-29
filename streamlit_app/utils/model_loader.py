
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_DIR / "models"

VGG16_PATH = MODELS_DIR / "vgg16_final_best.pth"
RESNET50_PATH = MODELS_DIR / "resnet50_final_best.pth"
MOBILENETV2_PATH = MODELS_DIR / "mobilenetv2_final_best.pth"
EFFICIENTNETB0_PATH = MODELS_DIR / "efficientnetb0_final_best.pth"

YOLO_PATH = (
    PROJECT_DIR
    / "runs"
    / "detection"
    / "yolov8s_v2_targeted_15ep"
    / "weights"
    / "best.pt"
)


# ============================================================
# DEVICE
# ============================================================

def get_device(prefer_gpu=True):

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


# ============================================================
# CHECKPOINT HELPER
# ============================================================

def _load_checkpoint(model, checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# ============================================================
# VGG16
# ============================================================

def load_vgg16(device):

    model = models.vgg16(weights=None)

    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(256, 25)
    )

    return _load_checkpoint(
        model,
        VGG16_PATH,
        device
    )


# ============================================================
# RESNET50
# ============================================================

def load_resnet50(device):

    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 25)
    )

    return _load_checkpoint(
        model,
        RESNET50_PATH,
        device
    )


# ============================================================
# MOBILENETV2
# ============================================================

def load_mobilenetv2(device):

    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 25)
    )

    return _load_checkpoint(
        model,
        MOBILENETV2_PATH,
        device
    )


# ============================================================
# EFFICIENTNET-B0
# ============================================================

def load_efficientnetb0(device):

    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),

        nn.Linear(1280, 512),

        nn.BatchNorm1d(512),

        nn.SiLU(),

        nn.Dropout(0.3),

        nn.Linear(512, 25)
    )

    return _load_checkpoint(
        model,
        EFFICIENTNETB0_PATH,
        device
    )


# ============================================================
# YOLOv8s
# ============================================================

def load_yolo():

    model = YOLO(
        str(YOLO_PATH)
    )

    return model


# ============================================================
# LOAD ALL CLASSIFIERS
# ============================================================

def load_all_classifiers(prefer_gpu=True):

    device = get_device(prefer_gpu)

    models_dict = {
        "VGG16": load_vgg16(device),
        "ResNet50": load_resnet50(device),
        "MobileNetV2": load_mobilenetv2(device),
        "EfficientNetB0": load_efficientnetb0(device)
    }

    return models_dict, device
