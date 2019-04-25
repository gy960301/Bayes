import torch
import torch.nn as nn
from mmodel.basic_module import WeightedModule
import numpy as np
import torch.nn.functional as F
from mground.gpu_utils import anpai

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class SGHMCLinear(WeightedModule):
    def __init__(self, in_features, out_features, param):
        # 继承父类属性
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rec_gsqr=rec_gsqr

        # 权重w的参数
        self.weight = nn.Parameter(torch.Tensor(out_features).random.randn()*init_sigma)
        
        # Bias 
        self.bias= nn.Parameter(
            torch.Tensor(out_features).zeros()
        )
        #gradient

    def forward(self, input, istrain=True):
        if istrain:
            weight=self.weight
            bias=self.bias

        return F.linear(input,weight,bias)

class SGHMCUpdater:
    def __init__()


class SGHMCNetwork(WeightedModule):
    def __init__(self,param):
        super().__init__()

        self.f1 = nn.Linear(28 * 28, 100,param)
        self.f2 = nn.Linear(100, 10,param)
        
        self.param=param

        self.relu_1 = nn.ReLU()
        

    def forward(self, x,sample=False):

        x = inputs.view(-1, 28 * 28)
        x = self.f1(x)
        x = self.relu_1(x)

        x = self.f2(x)
        x=F.softmax(x,dim=1)

        return x

    def update(self,xdata,ylabel):




    def predict(self,x,sample=True):
        p = self.param

        outputs = torch.zeros(samples, p.batch_size, p.class_num)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
        return outputs






