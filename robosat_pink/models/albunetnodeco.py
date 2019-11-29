import torch
import torch.nn as nn
from torchvision.models import resnet50


class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block."""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return nn.functional.relu(self.block(x), inplace=True)


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two."""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        return self.block(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


class AlbunetNODECO(nn.Module):
    def __init__(self, shape_in, shape_out, train_config=None):
        self.doc = """
            U-Net inspired encoder-decoder architecture with a ResNet encoder as proposed by Alexander Buslaev.

            - https://arxiv.org/abs/1505.04597 - U-Net: Convolutional Networks for Biomedical Image Segmentation
            - https://arxiv.org/pdf/1804.08024 - Angiodysplasia Detection and Localization Using DCNN
            - https://arxiv.org/abs/1806.00844 - TernausNetV2: Fully Convolutional Network for Instance Segmentation
        """
        self.version = 1

        num_filters = 32
        num_channels = shape_in[0]
        num_classes = shape_out[0]
        super().__init__()

        try:
            pretrained = train_config["model"]["pretrained"]
        except:
            pretrained = False
        pretrained=False
        self.resnet = resnet50(pretrained=pretrained,num_classes=num_classes)

        assert num_channels
        if num_channels != 3:
            weights = nn.init.xavier_uniform_(torch.zeros((64, num_channels, 7, 7)))
            if pretrained:
                for c in range(min(num_channels, 3)):
                    weights.data[:, c, :, :] = self.resnet.conv1.weight.data[:, c, :, :]
            self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.conv1.weight = nn.Parameter(weights)

    def forward(self,x):
        return self.resnet.forward(x)
