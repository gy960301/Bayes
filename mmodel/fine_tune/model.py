import itertools

import numpy as np
import torch

from mdata.partial_folder import MultiFolderDataHandler
from mground.gpu_utils import current_gpu_usage
from mground.math_utils import entropy, make_weighted_sum
from mtrain.mloger import GLOBAL, LogCapsule

from ..basic_module import DAModule
from .networks.networks import DomainClassifier
from ..utils.gradient_reverse import GradReverseLayer
from .params import get_params

param = get_params()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_scaler(
    iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75
):
    iter_num = iter_num - 1
    lr_scaler = 1 / (1 + 10 * iter_num / max_iter)**power
    return lr_scaler


class Finetune(DAModule):
    def __init__(self):
        super(Finetune, self).__init__(param)
        self._all_ready()

    def _regist_networks(self):

        if False:
            from .networks.resnet50 import ResFc, ResClassifer
            F = ResFc()
            C = ResClassifer(class_num=31)
        else:
            from .networks.alex import AlexNetFc, AlexClassifer

            F = AlexNetFc()
            C = AlexClassifer(class_num=31)

        return {"F": F, "C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": self.params.lr,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "nesterov": True,
            "lr_mult": {"F": 0.1},
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "lr_lambda": lambda steps: get_lr_scaler(
                steps, self.total_steps
            ),
            "last_epoch": 0,
        }

        self.define_loss(
            "global_looss",
            networks=["C"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

        self.define_log("classify")

    def _train_step(self, s_img, s_label, t_img):

        imgs = s_img

        backbone_feature = self.F(imgs)
        feature, pred_class = self.C(backbone_feature)

        loss_classify = self.ce(pred_class, s_label)


        self._update_loss("global_looss", loss_classify )
        self._update_logs({"classify": loss_classify})

        del loss_classify

    def _valid_step(self, img):
        feature = self.F(img)
        _, prediction = self.C(feature)
        return prediction
