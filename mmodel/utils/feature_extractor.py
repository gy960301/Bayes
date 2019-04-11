import torch
from torch import nn
from mmodel.basic_module import WeightedModule
from torchvision.models import resnet50

class GradReverseLayer(WeightedModule):
    def __init__(self, params, foctor=lambda: 1):
        self.has_init = True
        self.factor = foctor

    def forward(self, inputs):
        return inputs.view_as(inputs)

    def backward(self, grad_output):
        return grad_output * -self.factor()

class Res50FeatureExtroctor(WeightedModule):
    def __init__(self, params):
        super().__init__()

        res = resnet50(pretrained=True)
        layers = list(res.children())[:-2]

        self.F = torch.nn.Sequential(*layers)
        self.has_init = True

    def forward(self, inputs):
        features = self.F(inputs)
        return features

    def output_size(self):
        return [2048, 7, 7]
