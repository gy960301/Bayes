import torch
from torch import nn
from mmodel.basic_module import WeightedModule


def init_weights(m):
    classname = m.__class__.__name__
    if (
        classname.find("Conv2d") != -1
        or classname.find("ConvTranspose2d") != -1
    ):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class Net(WeightedModule):
    def __init__(self):
        super().__init__()

        self.f1 = nn.Linear(28 * 28, 400)
        self.f2 = nn.Linear(400, 400)
        self.f3 = nn.Linear(400, 10)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, inputs):

        x = inputs.view(-1, 28 * 28)
        x = self.f1(x)
        x = self.relu_1(x)

        x = self.f2(x)
        x = self.relu_2(x)

        x = self.f3(x)

        return x

