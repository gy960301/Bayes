"""this file gives the information needed to reconstruct official Caffe models in PyTorch"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),  # 0
            nn.ReLU(inplace=True),  # 1
            nn.Dropout(0.5),  # 2
            nn.Linear(4096, 4096),  # 3
            nn.ReLU(inplace=True),  # 4
            nn.Dropout(0.5),  # 5
            nn.Linear(4096, num_classes),  # 6
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = '/home/wogong/Models/pytorch/alexnet_cz2.pth'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model)
    return model


class AlexNetFc(nn.Module):
    """ AlexNet pretrained on imagenet for Office dataset"""

    def __init__(self):
        super(AlexNetFc, self).__init__()

        model_alexnet = alexnet(pretrained=True)

        self.features = model_alexnet.features

        self.fc = nn.Sequential()
        for i in range(6):
            self.fc.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features  # 


    def forward(self, input_data):
        feature = self.features(input_data)
        feature = feature.view(-1, 256 * 6 * 6)
        feature_output = self.fc(feature)
        return feature_output
