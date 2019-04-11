"""this file gives the information needed to reconstruct official Caffe models in PyTorch"""
import torch
import torch.nn as nn
from mmodel.basic_module import WeightedModule


def get_parameters(module, flag):
    """ flag = 'weight' or 'bias'
    """
    for name, param in module.named_parameters():
        if flag in name:
            yield param


class AlexNet(WeightedModule):
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
        model_path = "./_PUBLIC_DATASET_/alexnet_caffe.pth"
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model)
    return model

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class AlexNetFc(WeightedModule):
    """ AlexNet pretrained on imagenet for Office dataset"""

    def __init__(self):
        super(AlexNetFc, self).__init__()

        model_alexnet = alexnet(pretrained=True)

        self.features = model_alexnet.features

        self.fc = nn.Sequential()
        for i in range(7):
            self.fc.add_module(
                "classifier" + str(i), model_alexnet.classifier[i]
            )



        self.has_init = True

    def forward(self, input_data):
        feature = self.features(input_data)
        feature = feature.view(-1, 256 * 6 * 6)
        feature = self.fc(feature)
        return feature


def weights_init_helper(modul, params=None):
    """give a module, init it's weight
    
    Args:
        modul (nn.Module): torch module
        params (dict, optional): Defaults to None. not used.
    """
    import torch.nn.init as init

    for m in modul.children():
        # init Conv2d with norm
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            init.constant_(m.bias, 0)
        # init BatchNorm with norm and constant
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, 0)
        # init full connect norm
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            weights_init_helper(m)

        if isinstance(m, WeightedModule):
            m.has_init = True

class AlexGFC(WeightedModule):
    def __init__(self):
        super(AlexGFC, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(1000, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            # nn.Dropout(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            # nn.Dropout(),
        )

        weights_init_helper(self)
        self.has_init = True

    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature

class AlexClassifer(WeightedModule):
    def __init__(self, class_num, reversed_coeff):
        super(AlexClassifer, self).__init__()

        self.classifer = nn.Sequential(
            nn.Linear(100, class_num),
        )

        weights_init_helper(self)

        self.softmax = nn.Softmax(dim=1)

        self.class_num = class_num
        self.has_init = True

        assert callable(reversed_coeff)
        self.revgrad_hook = lambda grad: -reversed_coeff()*grad.clone()

    def forward(self, feature, adapt):

        if adapt:
            feature.register_hook(self.revgrad_hook)

        prediction_with_unkonw = self.classifer(feature)

        unkonw_prediction = self.softmax(prediction_with_unkonw)[:, -1].unsqueeze(1)

        return prediction_with_unkonw, unkonw_prediction

