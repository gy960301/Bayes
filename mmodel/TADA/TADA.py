import torch

from mtrain import mloger

from mground.math_utils import entropy

from ..params import get_params
from ..basic_module import DAModule
from ..utils.feature_extractor import Res50FeatureExtroctor


params = get_params()


def adversarial_losses(prediction, label):
    
    creterion = nn.BCELoss()

    label = label.repeat(prediction.size())

    loss_D = creterion(prediction, label)


class TADA(DAModule):
    def __init__(self):
        
        super(MANN, self).__init__(params)
        params.dataset = 'OFFICE'
        params.source = 'A'
        params.target = 'D'

        self.params = params

        self.spatial_size = 49

        ## losses criterion
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

        self._all_ready()

    def _regist_networks(self):
        networks = {
            "lD_" + str(i): DomainClassifier(input_dim=2048)
            for i in range(self.spatial_size)
        }

        networks['gD'] = DomainClassifier(input_dim=256)
        networks['F'] = FeatureExtroctor()
        networks['B'] = Bottleneck()
        return networks
    
    def _regist_losses(self):
        

        self.TrainCpasule.registe_default_optimer(
            torch.optim.SGD, lr=params.lr, momentum=0.95
        )
        self.define_loss('loss_F', self.F)

        self.define_loss('loss_B', self.B)
        self.define_loss('loss_gD', self.gD)
        lDs = [v for k, v in self.networks.items() if 'lD_' in k]
        self.define_loss('loss_lD', lDs)
    
    def _make_predicitons(self, img):

        def _local_dis(features):
            """ given a feature mask, producing it's local attention mask.
            
            Arguments:
                features {tensor} -- given feature
            
            Returns:
                tensor -- loal feature mask
            """
            size = features.size()
            batch = size[0]
            spatial_size = size[2] * size[3]

            ## OPTIMIZE features.view(batch, -1) ? 
            features = features.view(batch, -1, spatial_size)

            attens = [
                self.networks["l_D_" + str(i)](features[:, :, i])
                for i in range(spatial_size)
            ]

            return torch.cat(attens, dim=1)

        local_feature = self.F(s_img)

        ## calculate local attention based on l_atten = 2 - H(D(f))
        l_domain_predict = self._local_dis(features)
        l_atten = entropy(l_domain_predict, reduction='none')
        l_atten = 2 - l_atten  # 1 + (1 - l_atten)

        ## resize attention mask to size of feature
        size = features.size()
        l_atten = l_atten.view(size[0], size[2], size[3])
        l_atten = l_atten.unsqueeze(1)
        l_atten = l_atten.expand(size)

        ## reweight feature based on local attention
        features = features * l_atten

        ## go throught bottleneck layers
        g_features = self.B(features)

        ## cal global attention and make predictionp
        g_domain_predict = self.g_D(g_features)
        g_atten = entropy(g_domain_predict, reduction='none')
        g_atten = 1 + g_atten
    
        ## make prediction
        prediction = self.C(g_domain_predict)

        return l_domain_predict, g_domain_predict, prediction

    def _train_step(self, s_img, s_label, t_img):
        
        def calculate_losses(img, label=None):
            from_target = label is None

            ## make local domain, global domain and class prediction
            l_domain, g_domain, prediction, self._make_predicitons(img)
            
            domain = self.T if from_target else self.S
            domain = label.repeat(prediction.size())

            loss_lD = self.bce(l_domain, domain)
            loss_gD = self.bce(g_domain, domain)
            loss_dis = (loss_lD, loss_gD)

            ## NOTE the conf loss introduced in ADDA, min-log(D(F(xt)))
            loss_conf = None
            if from_target:
                loss_lConf = self.bce(l_domain, self.S)
                loss_gConf = self.bce(g_domain, self.S)
                loss_conf = (loss_lConf, loss_gConf)
                loss_prediction = entropy(prediction, reduction='mean')
            else:
                loss_prediction = self.ce(prediction, label)
            
            return loss_dis, loss_conf, loss_prediction
        
        S_loss_dis, _, S_loss_prediction = calculate_losses(s_img, s_label)
        T_loss_dis, l_conf, T_loss_prediction = calculate_losses(s_img, s_label)
    
