import torch
import torch.nn as nn
import torchvision.models as models


class VisualBackbone(nn.Module):
    def __init__(self, embed_size):
        super(VisualBackbone, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resent = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resent(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
