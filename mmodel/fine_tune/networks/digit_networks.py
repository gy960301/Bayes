from torch import nn
import torch
from mmodel.basic_module import WeightedModule

class FeatureExtractor(WeightedModule):
    def __init__(self, params):
        super().__init__()

        in_dim = 1 if params.gray else 3

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, inputs):
        feature = self.feature_conv(inputs)
        return feature
    
    def output_shape(self):
        return (64, 5, 5)

class Classifer(WeightedModule):

    def __init__(self, params, in_dim):
        super().__init__()
        self.classifer = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(1024, 64),
            nn.ReLU(True),

            nn.Linear(64, 10),
        )

    def forward(self, inputs):
        b = inputs.size()[0]
        predict = self.classifer(inputs.view(b, -1))
        return predict

class DomainClassifer(WeightedModule):

    def __init__(self, params, in_dim):
        super().__init__()
        self.classifer = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(512, 512),
            nn.ReLU(True),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, coeff=1):
        b = inputs.size()[0]
        inputs = inputs * 1
        if self.training:
            inputs.register_hook(lambda grad: grad.clone()*(-1)*coeff)
        predict = self.classifer(inputs.view(b, -1))
        return predict
