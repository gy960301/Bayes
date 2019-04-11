import logging

from torch import optim as optim
from torch import nn as nn
import torch

from mground.func_utlis import get_args_dict


class LossBuket(object):
    def __init__(self):
        self.value = None

    def loss(self):
        return self.value


class LossHolder(object):
    def __init__(self):
        self.loss_dic = dict()
        self.fixed = False

    def __getitem__(self, index):
        if self.fixed is True:
            return self.loss_dic[index]

        if not index in self.loss_dic:
            self.loss_dic[index] = LossBuket()
        return self.loss_dic[index]

    def fix_loss_keys(self):
        self.fixed = True


class TrainCapsule(nn.Module):
    """this is a tool class for helping training a network
    """

    # global optimer type and args
    __optim_type__ = None
    __optim_args__ = None

    # global decay op type and args
    __decay_op__ = None
    __decay_args__ = None

    # global decay op function
    ## NOTE this function will override decay op

    __recal_lr__ = None

    def __init__(
        self,
        optim_loss: LossBuket,
        optim_networks,
        optimer_info,
        decay_info,
        tagname=None,
    ):
        super(TrainCapsule, self).__init__()

        self.tag = tagname
        self.optim_loss = optim_loss

        # get all networks, and store them as list
        if not isinstance(optim_networks, (tuple, list)):
            networks_list = list()
            networks_list.append(optim_networks)
        else:
            networks_list = optim_networks
        self.optim_network = networks_list

        # get all parameters in network list
        self.all_params = list()
        optimer_type, optimer_kwargs = optimer_info

        base_lr = optimer_kwargs["lr"]
        base_decay = optimer_kwargs.get('weight_decay', 0)
        lr_mult_map = optimer_kwargs.get("lr_mult", dict())
        assert type(lr_mult_map) is dict


        for i in networks_list:
            if isinstance(i, torch.nn.DataParallel):
                i = i.module
            try:
                param_info = i.get_params()
                for p in param_info:
                    p['initial_lr'] = p.get('lr_mult', 1) * base_lr
                    p['weight_decay'] = p.get('decay_mult', 1) * base_decay
                    p['tag'] = i.tag
                
            except Exception:
                lr_mult = lr_mult_map.get(i.tag, 1)
                param_info = [{
                    "params": i.parameters(),
                    "lr_mult": lr_mult,
                    "lr": lr_mult * base_lr,
                    "initial_lr": lr_mult * base_lr,
                    'tag' : i.tag,
                },]
            self.all_params += param_info
        
        # init optimer base on type and args
        optimer_kwargs.pop("lr_mult", None)
        self.optimer = optimer_type(self.all_params, **optimer_kwargs)

        # init optimer decay option
        self.lr_scheduler = None
        if decay_info is not None:
            decay_type, decay_arg = decay_info
            self.lr_scheduler = decay_type(self.optimer, **decay_arg)

    def __all_networks_call(self, func_name):
        def __one_networkd_call(i):
            func = getattr(i, func_name)
            return func()

        map(__one_networkd_call, self.optim_network)

    def train_step(self, retain_graph=True):
        self.optimer.zero_grad()
        self.optim_loss.value.backward(retain_graph=True)
        self.optimer.step()

    def make_zero_grad(self):
        self.optimer.zero_grad()

    def decary_lr_rate(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


if __name__ == "__main__":

    pass

