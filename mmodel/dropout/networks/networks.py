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

class DropconNet(WeightedModule):
    def __init__(self, param):
        super().__init__()



        self.predictor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            
            nn.Dropout(p=0.3),

            nn.Linear(400, 400),
            nn.ReLU(),

            nn.Linear(400, 10),
        )

        # self.bn1 = nn.BatchNorm1d(400)
        # self.bn2 = nn.BatchNorm1d(400)

        # init_weights(self)

        # self.has_init = True


    def forward(self, inputs):

        x = inputs.view(-1, 28 * 28)
        x = self.predictor(x)
        # x=nn.log_softmax(x,dim=1)

        return x

