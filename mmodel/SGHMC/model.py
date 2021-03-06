import itertools

import numpy as np
import torch

from ..basic_module import TrainableModule, ELoaderIter
from .params import get_params
from .networks.sghmc_network import SGHMCNetwork

import mdata.dataloader as mdl
from mground.gpu_utils import anpai


# get params from defined basic params fucntion
param = get_params()


class SGHMCModel(TrainableModule):

    def __init__(self):
        super(SGHMCModel, self).__init__(param)

        # somethin you need, can be empty
        self.best_accurace = 0.0
        self.total = self.corret = 0
        self._all_ready()

    def _prepare_data(self):
        '''
            prepare your dataset here
            and return a iterator dic
        '''
        params = self.params

        train_set = mdl.get_dataset("MNIST", split="train")
        valid_set = mdl.get_dataset("MNIST", split="valid")

        def get_loader(dataset):
            l = torch.utils.data.DataLoader(
                dataset,
                batch_size=param.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=4,
            )
            return l

        train_loader = get_loader(train_set)
        valid_loader = get_loader(valid_set)


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
        net = SGHMCNetwork(param)
        return {"SGHMCN": net}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": param.lr,
            # "momentum": 0.9,
            #"weight_decay": 0.001,
            # "nesterov": True,
        }

        lr_scheduler = {
            "type": torch.optim.lr_scheduler.StepLR,
            "step_size": self.total_steps / 3,
            #"gamma":0.1
        }

        self.define_loss(
            "claasify_loss",
            networks=["SGHMCN"],
            optimer=optimer,
            #decay_op=lr_scheduler,
        )

        self.define_log("claasify", group="train")
        self.define_log("valid_loss", "valid_accu", group="valid")


    def _train_process(self, datas):

        img, label = datas

        predcition = self.SGHMCN(img)
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

