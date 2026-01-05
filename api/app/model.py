import torch
import torch.nn as nn
from torchvision import models

DEVICE = "cpu"

def load_model(weights_path: str):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

