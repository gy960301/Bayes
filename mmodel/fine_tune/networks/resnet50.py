import torch.nn as nn
from torchvision import models
from mmodel.basic_module import WeightedModule


class ResFc(WeightedModule):

    def __init__(self):
        super(ResFc, self).__init__()

        model_resnet50 = models.resnet50(pretrained=True)

        self.features = nn.Sequential(
            model_resnet50.conv1,
            model_resnet50.bn1,
            model_resnet50.relu,
            model_resnet50.maxpool,
            model_resnet50.layer1,
            model_resnet50.layer2,
            model_resnet50.layer3,
            model_resnet50.layer4,
            model_resnet50.avgpool,
        )

        self.has_init = True

    def forward(self, input):
        x = self.features(input)
        return x

class ResClassifer(WeightedModule):

    def __init__(self, class_num):
        super(ResClassifer, self).__init__()   

        classifer = nn.Linear(2048, class_num)

        nn.init.xavier_normal_(classifer.weight)
        nn.init.constant_(classifer.bias, 0)

        self.has_init = True

        self.classifer = classifer
    
    def forward(self, inputs):

        x = inputs.view(inputs.size(0), -1)
        prediction = self.classifer(x)
        return inputs, prediction

