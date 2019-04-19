import itertools

import numpy as np
import torch

from ..basic_module import TrainableModule, ELoaderIter
from .params import get_params
from .networks.bayes_network import BayesianNetwork

import mdata.dataloader as mdl
from mground.gpu_utils import anpai

param = get_params()

class BayesModel(TrainableModule):
    
    def __init__(self):
        super(BayesModel, self).__init__(param)

        self.best_accurace = 0.0
        self.total = self.corret = 0

        self._all_ready()

    def _prepare_data(self):

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

        params.len_train_batch = len(train_loader)
        params.len_test_batch = len(train_loader)

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
        net = BayesianNetwork(param)
        return {"BN": net}

    def _regist_losses(self):

        optimer = {
            "type": torch.optim.SGD,
            "lr": param.lr,
            # "momentum": 0.9,
            # "weight_decay": 0.001,
            # "nesterov": True,
        }

        # lr_scheduler = {
        #     "type": torch.optim.lr_scheduler.StepLR,
        #     "step_size": self.total_steps / 3,
        # }

        self.define_loss(
            "loss",
            networks=["BN"],
            optimer=optimer,
            # decay_op=lr_scheduler,
        )

        self.define_log("losses", group="train")
        self.define_log("valid_accu", group="valid")

    def _train_process(self, datas):

        img, label = datas

        loss, _, _, _ = self.BN.sample_elbo(img, label,param.trainsamples)

        self._update_loss("loss", loss)
        self._update_log("losses", loss)

    def _eval_process(self, datas):

        params = self.params

        end_epoch = datas is None

        def handle_datas(datas):

            img, target = datas
            # get result from a valid_step
            correct_count = 0
            corrects = torch.zeros(params.samples + 1)
            predicts=torch.zeros(params.samples+1,params.eval_batch_size,params.class_num)

            corrects, predicts = anpai((corrects, predicts), True, False)

            for i in range(params.samples):
                predicts[i]=self.BN(img,sample=True)
            predicts[params.samples]=self.BN(img,sample=False)
            predict=predicts.mean(0)
            

            preds = preds = predicts.max(2, keepdim=True)[1]
            pred = predict.max(1, keepdim=True)[1]  # 最大的log的索引
            corrects += (
                preds.eq(target.view_as(pred)).float().sum(dim=1).squeeze()
            )
            correct_count += pred.eq(target.view_as(pred)).sum().item()

            current_size = target.size()[0]

            for index, num in enumerate(corrects):
                if index == param.samples:
                    #print("Component{} Accurancy:{}/{}".format(index, num, current_size))
                
                    print("Posterior Mean Accurancy: {}/{}".format(num, current_size))
            

        
            #predict = self.BN(img)

            # calculate valid accurace and make record
            

            # pred_cls = predict.data.max(1)[1]
            # corrent_count = pred_cls.eq(label.data).sum()


            self._update_logs(
                {
                    "valid_accu": correct_count * 100 / current_size,
                },
                group="valid",
            )

            return correct_count, current_size

        if not end_epoch:
            right, size = handle_datas(datas)
            self.total += size
            self.corret += right
        else:
            accu = self.corret / self.total
            self.best_accurace = max((self.best_accurace, accu))
            self.total = 0
            self.corret = 0
