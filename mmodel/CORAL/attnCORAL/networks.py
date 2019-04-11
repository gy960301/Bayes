from torch import nn
import torch
from mmodel.basic_module import WeightedModule


class FeatureExtroctor(WeightedModule):
    def __init__(self, params):
        super().__init__()

        img_c = 1 if params.gray else 3

        self.feature_conv = nn.Sequential(
            nn.Conv2d(img_c, 64, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.feature_fc = nn.Sequential(
            nn.Linear(5*5*128, 1024),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
        )

    def forward(self, inputs):
        feature = self.feature_conv(inputs)
        b = feature.size()[0]
        feature = self.feature_fc(feature.view(b, -1))
        return feature

class Classifier(WeightedModule):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(True),
        )

        self.classify = nn.Sequential(
            nn.Linear(64, 10)
        )
    
    def forward(self, inputs):
        b = inputs.size()[0]
        feature = self.feature(feature.view(b, -1))
        predict = self.feature_classif(feature)
        return feature, predict

class DomainClassifier(WeightedModule):
    def __init__(self):
        super().__init__()

        self.classify = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            nn.Linear(64, 1)
        )


    def forward(self, inputs):
        b = inputs.size()[0]
        predict = self.classify(inputs.view(b, -1))
        return predict


    def get_output_shape(self):
        return (1024,1,1)

class Mask(WeightedModule):
    def __init__(self):
        super().__init__()

        self.mask = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),

            nn.Linear(512, 512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.ReLU(True),
        )
   
    def forward(self, inputs):
        b = inputs.size()[0]
        mask = self.mask(feature.view(b, -1))
        mask = self.cut(mask)
        return mask

    def cut(inputs):
        result = 1 / (1+50**(-100*inputs))

if __name__ == "__main__":

    x = torch.Tensor(3, 10, 5, 5).random_(0, 10)
    # spool = nn.AdaptiveAvgPool2d(1)

    # def cpool():
    #     pool = nn.AdaptiveAvgPool1d(1)
    #     b, c, w, h = x.size()
    #     x = x.view(b, c, w*h).permute(0, 2, 1)
    #     x = pool(x).view(b, 1, w, h)
    #     return x

    # print(cpool(x))
    # print(cpool(x).size())
