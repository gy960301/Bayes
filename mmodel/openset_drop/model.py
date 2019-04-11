import itertools

import numpy as np
import torch

from .networks.networks import DomainClassifier

from ..basic_module import DAModule, ELoaderIter
from .params import get_params


from mdata.partial.partial_dataset import require_openset_dataloader
from mdata.partial.partial_dataset import OFFICE_HOME_CLASS
from mdata.transfrom import get_transfrom
import numpy as np

param = get_params()

old_ne = 0
def norm_entropy(p, reduction="None"):
    p = p.detach()
    n = p.size()[1] - 1
    p = torch.split(p, n, dim=1)[0]
    p = torch.nn.Softmax(dim=1)(p)
    e = p * torch.log((p)) / np.log(n)
    ne = -torch.sum(e, dim=1)
    if reduction == "mean":
        ne = torch.mean(ne)
    elif reduction == "top5":
        ne, _ = torch.topk(ne, 5, dim=0, largest=False)
        ne = torch.mean(ne)
    elif reduction == "top5_m":
        global old_ne
        ne, _ = torch.topk(ne, 5, dim=0, largest=False)
        ne = torch.mean(ne)
        if old_ne == 0:
            old_ne = ne
        else:
            old_ne = 0.1*old_ne + 0.9*ne
        return old_ne
    return ne


def get_bias(iter_num, max_iter, high, alpha=30, center=0.1):
    zero_step = param.task_ajust_step + param.pre_adapt_step
    if iter_num < zero_step:
        return 0

    iter_num -= zero_step
    max_iter -= zero_step

    p = iter_num / max_iter

    z = (
        (
            1 / (1 + np.exp(-alpha * (p - center)))
            - 1 / (1 + np.exp(-alpha * (-center)))
        )
        * ((1 + np.exp(alpha * center)) / np.exp(alpha * center))
        * high
    )

    return z


def get_lambda(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):

    zero_step = param.task_ajust_step
    if iter_num < zero_step:
        return 0

    if iter_num < param.task_ajust_step + param.pre_adapt_step:

        iter_num -= zero_step
        max_iter = param.task_ajust_step + param.pre_adapt_step

        return np.float(
            2.0
            * (high - low)
            / (1.0 + np.exp(-alpha * iter_num / max_iter))
            - (high - low)
            + low
        )

    return 1


def get_lr_scaler(
    iter_num, max_iter, init_lr=param.lr, alpha=10, power=0.75
):

    if iter_num < param.task_ajust_step:
        return 1

    iter_num -= param.task_ajust_step
    max_iter -= param.task_ajust_step

    lr_scaler = np.float((1 + alpha * (iter_num / max_iter)) ** (-power))
    return lr_scaler


class OpensetDrop(DAModule):
    def __init__(self):
        super().__init__(param)

        # self.eval_after = int(0.15 * self.total_steps)

        self.early_stop = self.total_steps / 2

        source_class = set(OFFICE_HOME_CLASS[0:20])
        target_class = set(
            OFFICE_HOME_CLASS[0:20] + OFFICE_HOME_CLASS[40:65]
        )

        assert len(source_class.intersection(target_class)) == 20
        assert len(source_class) == 20 and len(target_class) == 45

        self.source_class = source_class
        self.target_class = target_class
        self.class_num = len(self.source_class) + 1

        self.element_bce = torch.nn.BCELoss(reduction="none")
        self.element_ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.DECISION_BOUNDARY = self.TARGET.fill_(1)

        self._all_ready()

    @property
    def dynamic_offset(self):
        high = param.dylr_high

        return get_bias(
            self.current_step,
            self.total_steps,
            high=high,
            alpha=param.dylr_alpht,
            center=param.dylr_center,
        )

    def _prepare_data(self):

        back_bone = "alexnet"
        source_ld, target_ld, valid_ld = require_openset_dataloader(
            source_class=self.source_class,
            target_class=self.target_class,
            train_transforms=get_transfrom(back_bone, is_train=True),
            valid_transform=get_transfrom(back_bone, is_train=False),
            params=self.params,
        )

        iters = {
            "train": {
                "S": ELoaderIter(source_ld),
                "T": ELoaderIter(target_ld),
            },
            "valid": ELoaderIter(valid_ld),
        }

        return None, iters

    def _regist_networks(self):

        if True:
            from .networks.alex import AlexNetFc, AlexGFC, AlexClassifer

            F = AlexNetFc()
            G = AlexGFC()
            C = AlexClassifer(
                class_num=self.class_num,
                # reversed_coeff=lambda: get_lambda(
                #     self.current_step, self.total_steps
                # ),
                reversed_coeff=lambda: 1,
            )

        return {"F": F, "G": G, "C": C}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": param.lr,
            "momentum": 0.9,
            "weight_decay": 0.001,
            # "nesterov": True,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.MultiStepLR,
            "gamma": 0.1,
            "milestones": [
                # param.task_ajust_step,
                # param.task_ajust_step + param.pre_adapt_step,
                (
                    (self.total_step / 3)
                    + param.task_ajust_step
                    + param.pre_adapt_step
                )
            ],
        }

        self.define_loss(
            "class_prediction",
            networks=["G", "C"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )
        self.define_loss(
            "domain_prediction",
            networks=["C"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )
        self.define_loss(
            "domain_adv",
            networks=["G"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

        self.define_log("valid_loss", "valid_accu", group="valid")
        self.define_log(
            "classify",
            "adv",
            "dis",
            "valid_data",
            "outlier_data",
            "drop",
            "tolorate",
            group="train",
        )

    def _train_step(self, s_img, s_label, t_img, t_label):

        if self.current_step == self.early_stop:
            raise Exception("early stop")

        g_source_feature = self.G(self.F(s_img))
        g_target_feature = self.G(self.F(t_img))

        s_predcition, _ = self.C(g_source_feature, adapt=False)
        t_prediction, t_domain = self.C(g_target_feature, adapt=True)

        loss_classify = self.ce(s_predcition, s_label)
        ew_dis_loss = self.element_bce(t_domain, self.DECISION_BOUNDARY)

        target_entropy = norm_entropy(t_prediction, reduction="none")
        if self.current_step > param.task_ajust_step:
            allowed_idx = (
                target_entropy
                - norm_entropy(
                    s_predcition, reduction=param.base_entropy_mode
                )
                < self.dynamic_offset
            )
        else:
            allowed_idx = (
                torch.abs(
                    target_entropy
                    - norm_entropy(
                        s_predcition, reduction=param.base_entropy_mode
                    )
                )
                < self.dynamic_offset
            )

        allowed_data_label = torch.masked_select(t_label, mask=allowed_idx)
        valid = torch.sum(allowed_data_label != self.class_num - 1)
        outlier = torch.sum(allowed_data_label == self.class_num - 1)
        drop = self.params.batch_size - valid - outlier

        allowed_idx = allowed_idx.float().unsqueeze(1)
        keep_prop = torch.sum(allowed_idx) / self.params.batch_size
        drop_prop = 1 - keep_prop
        dis_loss = torch.mean(ew_dis_loss * (1 - allowed_idx)) * drop_prop
        adv_loss = torch.mean(ew_dis_loss * allowed_idx) * keep_prop

        self._update_logs(
            {
                "classify": loss_classify,
                "dis": dis_loss,
                "adv": adv_loss,
                "valid_data": valid.float(),
                "outlier_data": outlier.float(),
                "drop": drop.float(),
                "tolorate": self.dynamic_offset,
            }
        )

        if keep_prop != 0:
            self._update_losses(
                {
                    "class_prediction": loss_classify,
                    "domain_prediction": dis_loss + adv_loss,
                    "domain_adv": adv_loss / keep_prop,
                }
            )
        else:
            self._update_losses({"class_prediction": loss_classify})

        del loss_classify, adv_loss

    def _valid_step(self, img):
        feature = self.G(self.F(img))
        prediction, _ = self.C(feature, adapt=False)
        return prediction
