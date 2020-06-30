import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, embed_size):
        super(ResNet50, self).__init__()
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


class ResNet101(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(ResNet101, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet101(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

    def forward(self, images):

        with torch.no_grad():
            # (batch_size, 2048, image_size/32, image_size/32)
            out = self.resnet(images)

        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)

        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out
