import itertools

import numpy as np
import torch

from mdata.partial_folder import MultiFolderDataHandler
from mground.gpu_utils import current_gpu_usage
from mground.math_utils import entropy, make_weighted_sum
from mtrain.mloger import GLOBAL, LogCapsule, TRAIN, VALID, BUILD, HINTS

from ..basic_module import TrainableModule, ELoaderIter, logger
from .params import get_params
from .networks.networks import Net
from mground.log_utils import tabulate_log_losses


import mdata.dataloader as mdl


# get params from defined basic params fucntion
param = get_params()


class NewModel(TrainableModule):

    def __init__(self):
        super(BayesModel, self).__init__(param)

        # somethin you need, can be empty
        self._all_ready()

    def _prepare_data(self):
        '''
            prepare your dataset here
            and return a iterator dic
        '''

        train_set = None
        valid_set = None

        iters = {
            "train": ELoaderIter(train_loader),
            "valid": ELoaderIter(valid_loader),
        }

        return None, iters

    def _feed_data(self, mode, *args, **kwargs):

        assert mode in ["train", "valid"]

        its = self.iters[mode]
        if mode == "train":
            return its.next()
        else:
            return its.next(need_end=True)

    def _regist_networks(self):
        net = Net()
        return {"N": net}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.01,
            # "momentum": 0.9,
            "weight_decay": 0.001,
            # "nesterov": True,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.StepLR,
            "step_size": self.total_steps / 3,
        }

        self.define_loss(
            "claasify_loss",
            networks=["N"],
            optimer=optimer,
            decay_op=lr_scheduler,
        )

        self.define_log("claasify", group="train")
        self.define_log("valid_loss", "valid_accu", group="valid")


    def _train_process(self, datas):

        img, label = datas

        predcition = self.N(img)
        loss = self.ce(predcition, label)

        self._update_loss("claasify_loss", loss)
        self._update_log("claasify", loss)

    def _log_process(self):

        losses = [
            (k, v.log_current_avg_loss(self.current_step + 1))
            for k, v in self.train_loggers.items()
        ]

        return losses

    def _eval_process(self, datas):

        params = self.params

        end_epoch = datas is None

        def handle_datas(datas):

            img, label = datas
            # get result from a valid_step
            predict = self.N(img)

            # calculate valid accurace and make record
            current_size = label.size()[0]

            # pred_cls = predict.data.max(1)[1]
            # corrent_count = pred_cls.eq(label.data).sum()

            _, predic_class = torch.max(predict, 1)
            corrent_count = (
                (torch.squeeze(predic_class) == label).sum().float()
            )

            self._update_logs(
                {
                    "valid_loss": self.ce(predict, label),
                    "valid_accu": corrent_count * 100 / current_size,
                },
                group="valid",
            )

            return corrent_count, current_size

        if not end_epoch:
            right, size = handle_datas(datas)
            self.total += size
            self.corret += right
        else:
            logger.log(VALID, "End a evaling step.")
            accu = self.corret / self.total
            self.best_accurace = max((self.best_accurace, accu))
            self.total = 0
            self.corret = 0

