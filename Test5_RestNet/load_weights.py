import torch
import os
from model import resnet34
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weights_path = 'Test5_RestNet/weights/resnet34-333f7ec4.pth'

assert os.path.exists(model_weights_path), f"Model weights not found at {model_weights_path}"

net = resnet34()
net.load_state_dict(torch.load(model_weights_path, map_location=device))

in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel,5)

