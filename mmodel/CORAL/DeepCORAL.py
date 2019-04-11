from mmodel.basic_module import DAModule
from mmodel.CORAL.deepCORAL.networks import *
from mmodel.CORAL.coral import loss_CORAL


import torch
from params import get_param_parser
from mmodel.mloger import GLOBAL
import logging


def get_coral_param():
    parser = get_param_parser()

    parser.add_argument(
        "--coral_param", type=int, default=8, help="Use GPU to train the model"
    )

    return parser.parse_args()


class DeepCORAL(DAModule):
    def __init__(self, params):
        super(DeepCORAL, self).__init__(params)

    def _regist_networks(self):
        C = Classifier(self.params)
        return {"C": C}

    def _regist_losses(self):
        # set default optim function
        self.TrainCpasule.registe_default_optimer(
            torch.optim.Adam, lr=params.lr
        )
        self.define_loss("predict", "C")

    def _train_step(self, s_img, s_label, t_img):

        s_feature, s_predict = self.networks['C'](s_img)
        t_feature, t_predict = self.networks['C'](t_img)

        l_classifer = self.ce(s_predict, s_label)
        l_coral = loss_CORAL(s_feature, t_feature)

        self._update_loss("predict", l_classifer + self.params.coral_param * l_coral)

    def _valid_step(self, img):
        _, predict = self.C(img)
        return predict


if __name__ == "__main__":

    params = get_coral_param()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, format=" \t | %(levelname)s |==> %(message)s"
    )

    coral = DeepCORAL(params)
    coral.train_module()

    # from torchvision import models
    # from torchsummary import summary

    # feature = Classifier()
    # feature.weight_init()
    # summary(feature, (3, 32, 32))

