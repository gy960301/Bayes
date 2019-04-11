from torch import nn
import torch
from mmodel.basic_module import WeightedModule


from torchvision.models import resnet50

class Bottleneck(WeightedModule):

    def __init__(self, params):
        super().__init__()

        bottleneck_dim = 256
        bottleneck_linear = nn.Linear(2048*49, bottleneck_dim)
        bottleneck_linear.weight.data.normal_(0, 0.005)
        bottleneck_linear.bias.data.fill_(0.0)

        self.bottleneck = nn.Sequential(
            bottleneck_linear,
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(True),
        )
    
    def forward(self, inputs):
        b = inputs.size()[0] # [1, 2048, 7, 7]
        inputs = inputs.view(b, -1)
        features = self.bottleneck(inputs)
        return features

    def output_size(self):
        return [2048, 7, 7]

class Classifier(WeightedModule):
    def __init__(self, class_num):
        super().__init__()
        self.fc = nn.Linear(256, class_num)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)
        self.has_init = True

    def forward(self, inputs):
        predict = self.fc(inputs)
        return predict

class DomainClassifier(WeightedModule):
    def __init__(self, input_dim = 2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,1024)
        self.layer2 = nn.Linear(1024,1024)
        self.layer3 = nn.Linear(1024,1)

        self.layer1.weight.data.normal_(0, 0.01)
        self.layer2.weight.data.normal_(0, 0.01)
        self.layer3.weight.data.normal_(0, 0.3)

        self.layer1.bias.data.fill_(0.0)
        self.layer2.bias.data.fill_(0.0)
        self.layer3.bias.data.fill_(0.0)      

        self.droupout1 = nn.Dropout(0.5)
        self.droupout2 = nn.Dropout(0.5)
        
        self.relu1 = nn.LeakyReLU()  
        self.relu2 = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.has_init = True

    def forward(self, inputs):
        b = inputs.size()[0]
        x = inputs.view(b,-1)

        x = self.layer1(x)
        x = self.relu1(x)
        x = self.droupout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.droupout2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

    def get_output_shape(self):
        return (1024,1,1)


if __name__ == "__main__":

    x = torch.Tensor(3, 10, 5, 5).random_(0, 10)
    # spool = nn.AdaptiveAvgPool2d(1)
    # def cpool():
    #     pool = nn.AdaptiveAvgPool1d(1)
    #     b, c, w, h = x.size()
    #     x = x.view(b, c, w*h).permute(0, 2, 1)
    #     x = pool(x).view(b, 1, w, h)
    #     return x


    # print(cpool(x).size())
