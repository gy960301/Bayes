import itertools
import logging

import torch

from mdata.partial_folder import MultiFolderDataHandler
from mground.gpu_utils import current_gpu_usage
from mground.math_utils import entropy, make_weighted_sum
from mtrain.mloger import GLOBAL, LogCapsule

from ..basic_module import *
from ..utils.feature_extractor import Res50FeatureExtroctor
from .networks import *
from .params import get_params

params = get_params()
STAGE = {"adaptation": 0, "prediction": 1, "training": 3}


class Domain(object):
    T = 0
    S = 1


class PredictUnit(TrainableModule):
    def __init__(
        self, turn_key, hold_key, classes_map, params, bottleneck_layer
    ):
        super(PredictUnit, self).__init__(params)

        self.use_local_atten = params.local_attention
        self.params = params

        self.B = bottleneck_layer

        self._turn_key = turn_key
        self.hold_key = hold_key

        self.classes_to_idx = classes_map

        self.output_shape = None
        self.datas = None

        self.total_steps = 1
        self.eval_step = -1
        self.eval_once = True

        self.bceloss = torch.nn.BCELoss()
        self.local_bceloss = torch.nn.BCELoss(reduction="none")

        self.spital_size = 49

        self.g_dis_losses = list()

        ## NOTE checking important attribute.
        assert self.eval_once == True

        self._all_ready()

    @property
    def turn_key(self):
        return self._turn_key[0]

    def _prepare_data(self):
        return None, None

    def set_data(self, inputs):
        stage, datas = inputs
        self.current_stage = stage
        self.current_data = datas

    def _feed_data(self, mode):

        stage = self.current_stage

        if stage == STAGE["adaptation"]:
            feature, domain = self.current_data
            if domain is None:
                return feature
            else:
                return feature, domain

        elif stage == STAGE["prediction"]:
            g_feature = self.current_data
            return g_feature

        elif stage == STAGE["training"]:
            predict_loss = self.current_data
            return predict_loss
        else:
            raise Exception("Stage Error")

    def _regist_networks(self):

        regist_dict = {}
        if self.use_local_atten:
            regist_dict = {
                "l_D_" + str(i): DomainClassifier(input_dim=2048)
                for i in range(self.spital_size)
            }

        regist_dict["C"] = Classifier(len(self.classes_to_idx))

        regist_dict["g_D"] = DomainClassifier(input_dim=256)

        return regist_dict

    def _regist_losses(self):

        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=self.params.lr, momentum=0.95
        )

        if self.use_local_atten:
            local_dis_nets = [
                "l_D_" + str(i) for i in range(self.spital_size)
            ]
            self.define_loss("local_dis", *local_dis_nets)

        self.define_loss("global_dis", "g_D")

        self.define_loss("predict", "C")

    def _local_dis(self, features):
        """ given a feature mask, producing it's local attention mask.
        
        Arguments:
            features {tensor} -- given feature
        
        Returns:
            tensor -- loal feature mask
        """

        # [batch_size, c, h , w]
        size = features.size()
        # [batch_szie, h * w]
        batch = size[0]
        spatial_size = size[2] * size[3]
        features = features.view(batch, -1, spatial_size)

        result = [
            self.networks["l_D_" + str(i)](features[:, :, i])
            for i in range(spatial_size)
        ]

        result = torch.cat(result, dim=1)

        return result

    def _stage(self, stage, datas, is_training):

        assert stage in STAGE.values()

        #########################################
        ## begain stage one
        ## in the stage one local_atten and global atten are calculated
        #########################################
        if stage == STAGE["adaptation"]:

            if is_training:
                features, domain_label = datas
            else:
                features = datas

            l_atten = None
            if self.use_local_atten:
                # calculate local attention based on l_atten = 2 - H(D(f))
                l_domain_predict = self._local_dis(features)
                l_atten = entropy(l_domain_predict.detach())
                l_atten = 2 - l_atten  # 1 + (1 - l_atten)

                # resize to size of feature
                size = features.size()
                l_atten = l_atten.view(size[0], size[2], size[3])
                l_atten = l_atten.unsqueeze(1)
                l_atten = l_atten.expand(size)

                # reweight feature based on local attention
                features = features * l_atten

            # go throught bottleneck layers
            g_features = self.B(features)

            ## NOTE the attention value will be calculated without grad.
            # cal global attention and make predictionp
            g_domain_predict = self.g_D(g_features)
            g_atten = entropy(g_domain_predict.detach())
            g_atten = 1 + g_atten

            ## the grad will be compute.
            ## here l_atten and g_atten's requires_grad = True
            self.attentions = (l_atten, g_atten)

            #! If data comes from current unit then need trainning
            # update local domain discriminator
            # and global domain discriminator
            if self.turn_key == self.hold_key and is_training:

                l_diss = l_confuse = None
                if self.use_local_atten:
                    # resize local predict result to [b*49, 1]
                    l_domain_predict = l_domain_predict.view(-1, 1)
                    l_diss, l_confuse = self._adversiral_loss(
                        l_domain_predict, domain_label
                    )
                # perform the same process for global dis as local dis
                g_diss, g_confuse = self._adversiral_loss(
                    g_domain_predict, domain_label
                )

                return (l_diss, g_diss), (l_confuse, g_confuse)
            return None

        #########################################
        ## begain stage two
        ## in the stage tow, classes prediction are made
        #########################################
        elif stage == STAGE["prediction"]:
            ## NOTE g_features has enhanced in union
            g_features = datas
            # retrieval previous features
            predict_result = self.C(g_features)
            self.predict = predict_result

        #########################################
        ## begain stage two
        ## in the stage tow, classes prediction are made
        #########################################
        elif stage == STAGE["training"]:
            assert is_training
            predict_loss = datas
            g_diss = sum(self.g_dis_losses)
            print(g_diss)

            ## REVIEW here constant 49 revers the 'average' op when calculate it.
            ## WARM here the constance revers opration should be cheacked
            if self.use_local_atten:
                self._update_loss("local_dis", l_diss * self.spital_size)
            self._update_loss("global_dis", g_diss)
            # self._update_loss("predict", predict_loss)

            self.g_dis_losses.clear()
            del g_diss

        else:
            raise Exception("stage false." + str(stage))

    def _train_process(self, datas, **kwargs):
        stage = self.current_stage

        if stage == STAGE["adaptation"]:
            if self.turn_key == self.hold_key:
                dis, conf = self._stage(stage, datas, is_training=True)
                self.g_dis_losses.append(dis[1])
                self.confuse_losses = conf
            else:
                self._stage(stage, datas, is_training=True)

        elif stage == STAGE["prediction"]:
            self._stage(stage, datas, is_training=True)

        elif stage == STAGE["training"]:
            self._stage(stage, datas, is_training=True)

        else:
            raise Exception("No such stage!")

    def _finish_a_train_process(self):
        if self.current_stage == STAGE["training"]:
            self.current_step += 1

    def _log_process(self):
        return

    def _eval_process(self, datas, **kwargs):
        stage = self.current_stage
        self._stage(stage, datas, is_training=False)

    def _adversiral_loss(self, prediton, original_domain):
        creterion = self.bceloss

        original_domain = original_domain.repeat(prediton.size())

        loss_D = creterion(prediton, original_domain)

        # original gan loss will be:
        # loss_G = bce_loss(source_predict, TARGET_DOMAIN)
        # loss_G = bce_loss(target_predict, SOURCE_DOMAIN)
        target = torch.abs(original_domain - 0.5)
        loss_G = creterion(prediton, target) 

        return loss_D, loss_G

    def __str__(self):
        return "Unit of %s in group %d " % (self.domain, self.idx)


class Network(TrainableModule):
    def __init__(self):
        super(Network, self).__init__(get_params())
        params = get_params()

        self.use_local_atten = params.local_attention
        # init multi source image folder handler
        mhandler = MultiFolderDataHandler(
            root="Office_Shift",
            sources=["A", "W"],
            target="D",
            params=params,
        )

        # get all class separations
        # ics is a list of ClassSeperation
        # every ClassSeperation is consist of (classes) and (domains)
        ics = mhandler.independ_class_seperation

        unit_order = [
            (idx, domain)
            for idx, sep in enumerate(ics)
            for domain in sep.domains
        ]

        self.mhandler = mhandler
        self.independ_class_seperation = ics
        self.unit_order = unit_order

        # set group iterator
        class_group_idx = [i for i in range(len(ics))]
        self.group_idx_iter = itertools.cycle(iter(class_group_idx))
        self.class_group_idxs = class_group_idx

        self.CE = nn.CrossEntropyLoss()

        self._turn_key = [None]

        # self.valid_loss_logger = LogCapsule(
        #     self.losses['valid_loss'], 'valid_loss', to_file= False
        # )

        self._all_ready()

        self.accur_logger = LogCapsule(
            self.losses["valid_accu"], "valid_accu", to_file=False
        )

        self.eval_right = list()
        self.eval_count = list()
        self.best_accurace = 0

    @property
    def turn_key(self):
        return self._turn_key[0]

    @turn_key.setter
    def turn_key(self, turn_key):
        # assert len(turn_key) == 2
        self._turn_key[0] = turn_key

    def _iter_all_unit(self, func, **kwargs):
        for idx, domain in self.unit_order:
            func(idx, domain, **kwargs)

    def _regist_networks(self):

        #########################################
        ## two stage featrue extractor
        #########################################
        bottleneck = Bottleneck(self.params)
        F = Res50FeatureExtroctor(self.params)

        #########################################
        ## prediction union and units
        #########################################
        classes_sep = [
            i.classes_to_idx for i in self.independ_class_seperation
        ]
        unions = [{} for i in range(len(self.independ_class_seperation))]

        def create_predict_unit(idx, domain):
            hold_key = (idx, domain)
            p = PredictUnit(
                self._turn_key,
                hold_key,
                classes_sep[idx],
                self.params,
                bottleneck,
            )
            unions[idx][domain] = p

        self._iter_all_unit(create_predict_unit)
        self.unions = unions

        classifer = list()
        discriminator = list()

        def get_classifer(idx, domain):
            p = self.unions[idx][domain]
            classifer.append(p.networks["C"])
            discriminator.append(p.networks["g_D"])

        self.classifers = classifer
        self.discriminators = discriminator

        return {"F": F, "B": bottleneck, "softmax": nn.Softmax(dim=0)}

    def _regist_losses(self):

        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=self.params.lr, momentum=0.95
        )
        self.define_loss(
            "loss",
            self.networks["B"],
            self.networks["F"],
            # *self.classifers,
        )

        self.define_log("s_prediction")
        self.define_log("t_prediction")
        self.define_log("s_confusion")
        self.define_log("t_confusion")

    def _prepare_data(self):
        """ return dataloader.
        
        Returns:
            list -- datainfo, train_loader, valid_loader

            datainfo is dict conatins data info pairs

            train_loader is a tuple:
                first item is source data loaders whic is a dict of
                form {group_idx : {domain : dataloader}}

                second item is dataloader of target domain.
            
            valid loader is same as target train loader.
        """

        data_info = None

        ## NOTE drop last batch here
        icgs, target_loader, valid_loader = self.mhandler.seperation_with_loader(
            drop_last=True
        )

        # return all iters
        iters = dict()

        # get dataloader in orders
        partial_source_loaders = [
            {} for i in range(len(self.independ_class_seperation))
        ]

        def unit_loader(idx, domain):
            l = icgs[idx].get_domain_loader(domain)
            it = ELoaderIter(l)
            partial_source_loaders[idx][domain] = it

        self._iter_all_unit(unit_loader)

        # set trainging iters
        mode = "train"
        iters[mode] = dict()
        iters[mode]["target"] = ELoaderIter(target_loader)
        iters[mode]["source"] = partial_source_loaders

        # set validing iters
        mode = "valid"
        loader = valid_loader
        iters[mode] = ELoaderIter(loader)

        return data_info, iters

    def _feed_data(self, mode):

        if mode == "train":
            iters = self.iters[mode]["source"]
            target_iters = self.iters[mode]["target"]

            feed_datas = list()

            def feeded_datas(idx, domain):
                img, label = iters[idx][domain].next()
                feed_datas.append(img)
                feed_datas.append(label)

            self._iter_all_unit(feeded_datas)

            feed_datas.append(target_iters.next()[0])
            return feed_datas
        else:
            return self.iters[mode].next(need_end=True)

    def _train_process(self, datas, **kwargs):

        ## NOTE in a trainning process, multi batch will be feed
        source_datas = [
            (datas[i], datas[i + 1])
            for i in range(0, len(self.unit_order) * 2, 2)
        ]
        target_data = ((datas[-1], None),)

        def get_losses_from(datas, domain):
            """ based on feeded 'datas' and correspond 'domain' calculating 
            'confusion loss' and 'prediction loss'. When datas from target domain, 
            prediction loss will be the entropy loss of prediction.
            
            Arguments:
                datas {list} -- A list of data tuple, which contain img and label
                domain {int} -- 1 and 0 for source and target domain.
            
            Returns:
                list -- a list conatins 3 items: 'local confusion loss', 
                'global confusion loss' and 'prediction loss'.
            """

            #########################################
            #! Calculate prediction losses
            ## Iter all feed data, with the help of predict union
            ## make a prediction, and calculate predict_loss
            #########################################
            predict_losses = list()

            for idx, (img, label) in enumerate(datas):

                # get source features
                feature = self.F(img)

                #########################################
                #! Update turn_key
                #! If len of unions is 3,then trun_key will change from 0-2
                ##
                ## Notice that all unit will get the key only ONCE
                ## in this batch of data
                #########################################
                sep_id, sdomain = self.unit_order[idx]
                self.turn_key = (sep_id, sdomain)

                #########################################
                #! make prediction
                #########################################
                final_predict = self._make_prediction(feature, domain)


                #########################################
                #! calculate predicton losses
                #########################################
                if label is not None:
                    predict_loss = self.CE(final_predict, label)
                else:
                    predict_loss = entropy(
                        final_predict, reduction="sum", binary=False
                    )
                predict_losses.append(predict_loss)

            assert len(predict_losses) == len(datas)

            #########################################
            #! Get confusion loss
            #########################################

            g_confusion_losses = list()
            l_confusion_losses = list()

            def get_confusion_loss(idx, domain):
                c = self.unions[idx][domain].confuse_losses
                l_confusion_losses.append(c[0])
                g_confusion_losses.append(c[1])

            self._iter_all_unit(get_confusion_loss)

            l_confusion_loss = (
                torch.mean(torch.stack(l_confusion_losses))
                if self.use_local_atten
                else None
            )
            g_confusion_loss = torch.mean(torch.stack(g_confusion_losses))
            predict_loss = torch.mean(torch.stack(predict_losses))

            return l_confusion_loss, g_confusion_loss, predict_loss

        s_lconf, s_gconf, s_predict = get_losses_from(
            source_datas, Domain.S
        )

        t_lconf, t_gconf, t_predict = get_losses_from(
            target_data, Domain.T
        )

        predict_loss = s_predict + self.params.gamma * t_predict
        confuse_loss = s_gconf + t_gconf

        inputs = (STAGE["training"], predict_loss)

        def train_unit(idx, domain):
            u = self.unions[idx][domain]
            u.set_data(inputs)
            u.train_module()

        self._iter_all_unit(train_unit)

        self._update_log("s_prediction", s_predict)
        self._update_log("t_prediction", t_predict)
        self._update_log("s_confusion", s_gconf)
        self._update_log("t_confusion", t_gconf)

        self._update_loss(
            "loss", confuse_loss, retain_graph=False
        )

        ## OPTIMIZE delete varilable
        del datas, predict_loss, confuse_loss

    def _make_prediction(self, feature, from_domain=None):

        #########################################
        ## If domain is provided then the process
        ## is trainning else the evaling
        #########################################
        v = from_domain == None
        if from_domain is not None:
            func_name = "train_module"
            domain_label = torch.Tensor(1, 1).fill_(from_domain)
        else:
            func_name = "eval_module"
            domain_label = None
            from_domain = Domain.T

        def unit_process(unit: PredictUnit):
            f = getattr(unit, func_name)
            f()

        union_range = range(len(self.independ_class_seperation))

        #########################################
        #! perfoorming adaptation stage
        ##
        ## In this stage, all attention will be calculated.
        ##
        ## Loss function will be saved in unit when the key is
        ## right.
        #########################################
        inputs = (STAGE["adaptation"], (feature, domain_label))
        l_attens = [{} for i in union_range]
        g_attens = [{} for i in union_range]

        def stage_adaptation(idx, domain):

            unit = self.unions[idx][domain]
            # Domain label indicate that currennt process is
            # trainning or evaling.
            if from_domain == Domain.T:
                self.turn_key = unit.hold_key
            unit.set_data(inputs)
            unit_process(unit)
            l, g = unit.attentions
            l_attens[idx][domain] = l
            g_attens[idx][domain] = g

        self._iter_all_unit(stage_adaptation)

        #########################################
        #! perfoorming classify stage
        ##
        ## In this stage, new local attention and original feature
        ## map are feeded into unit
        ##
        ## the classify result will be calculated
        #########################################

        ## calculate union local attention and global attention
        def make_union_attention(tensors):
            l = torch.stack(tensors)
            return torch.mean(l, dim=0)

        union_g_atten = list(
            map(
                make_union_attention,
                [list(g_attens[i].values()) for i in union_range],
            )
        )

        if self.use_local_atten:

            union_l_atten = list(
                map(
                    make_union_attention,
                    [list(l_attens[i].values()) for i in union_range],
                )
            )

            #! Enhance Local Feature based on Union Local Attention
            enhanced_global_feature = [
                self.B(feature * union_l_atten[i])
                for i in range(len(union_range))
            ]

        else:
            enhanced_global_feature = self.B(feature)

        ## make prediction from all unit
        predict = [{} for i in union_range]

        def stage_classify(idx, domain):
            inputs = (STAGE["prediction"], enhanced_global_feature)
            unit = self.unions[idx][domain]
            unit.set_data(inputs)
            unit_process(unit)
            predict[idx][domain] = unit.predict

        self._iter_all_unit(stage_classify)

        ## based on unit result, construct union prediction result.
        union_predict = list(
            map(
                make_weighted_sum,
                [
                    (list(predict[i].values()), list(g_attens[i].values()))
                    for i in union_range
                ],
            )
        )

        final_predict = torch.cat(
            [union_predict[i] * union_g_atten[i] for i in union_range],
            dim=1,
        )

        if self.current_step % 10 ==0:
            print(str(self.current_step) + '===========================')
            for i in union_range:
                print(union_g_atten[i][0].item())

        final_predict = self.softmax(final_predict)

        return final_predict

    def _eval_process(self, datas, **kwargs):
        ## WARM this process is assert self.eval_ones == True
        def eval(img, label):
            predict = self._make_prediction(feature=self.F(img))

            _, predict_idx = torch.max(predict, 1)
            correct = (predict_idx == label).sum().float()
            return correct.cpu().numpy()

        need_train = datas is not None

        if need_train:
            total = len(self.mhandler.target_set)
            img, label = datas
            count = img.size()[0]
            correct = eval(img, label)
            self.eval_count.append(count)
            self.eval_right.append(correct)

        else:
            percent = sum(self.eval_right) / sum(self.eval_count)
            self.accur_logger.update_record(percent)
            self.accur_logger.log_current_avg_loss(self.current_step)
            self.eval_count.clear()
            self.eval_count.clear()
            self.best_accurace = max([self.best_accurace, percent])

    def _log_process(self):

        logging.info(
            "Steps %3d ends. Remain %3d steps to go. Fished %.2f%%"
            % (
                self.current_step + 1,
                self.params.steps - self.current_step - 1,
                (self.current_step + 1) / (self.params.steps + 1) * 100,
            )
        )

        logging.info(
            "Current best accurace is %3.3f%%."
            % (self.best_accurace * 100)
        )

        for v in self.train_loggers.values():
            v.log_current_avg_loss(self.current_step + 1)


if __name__ == "__main__":

    params = get_c_param()

    GLOBAL._TAG_ = params.tag

    logging.basicConfig(
        level=logging.INFO, format=" \t | %(levelname)s |==> %(message)s"
    )

    n = Network(params)

    try:
        n.train_module()
    finally:
        current_gpu_usage()
