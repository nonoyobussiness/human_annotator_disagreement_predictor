import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class DissagreementPredictor(nn.Module):
    def __init__(self, head='linear', pretrained=None, num_classes=10):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None
        base = models.resnet18(weights=weights)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feat_dim = 512

        if head == 'linear':
            self.head = nn.Linear(feat_dim, num_classes)

        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        elif head == 'temperature':
            self.head = nn.Linear(feat_dim, num_classes)
            self.temperature = nn.Parameter(torch.ones(1))

        else:
            raise ValueError(f"Unsupported head type: {head}")

        self.head_type = head

    def forward(self, x):
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        logits = self.head(feat)

        if self.head_type == 'temperature':
            logits = logits / self.temperature.clamp(min=0.1)

        return torch.softmax(logits, dim=-1)
