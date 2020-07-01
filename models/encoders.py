import torch
import torch.nn as nn
import torchvision.models as models


class EncoderNormal(nn.Module):
    def __init__(self, embed_size):
        super(EncoderNormal, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resent = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        with torch.no_grad():
            # (batch_size, 2048, 1, 1)
            features = self.resent(images)

        # (batch_size, 2048)
        features = features.view(features.size(0), -1)

        # (batch_size, embed_size)
        features = self.embed(features)
        return features


class EncoderAttention(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderAttention, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):

        with torch.no_grad():
            # (batch_size, 2048, image_size/32, image_size/32)
            out = self.resnet(images)

        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.embed(out)

        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out
