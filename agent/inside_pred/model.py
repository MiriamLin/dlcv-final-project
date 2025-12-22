import torch
from torchvision import models
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_inside_model(model_cfg):
    in_channels = model_cfg.get('input_channels', 5)

    # load model from path
    if 'model_path' in model_cfg:
        model_path = model_cfg['model_path']
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        model = ResNet50Binary(in_channels=in_channels)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
    
    return model

class ResNet50Binary(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Change input conv layer to accept in_channels
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Binary classification head
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze(1)  # [B] logits