import torch
import torch.nn as nn
from mmodel.basic_module import WeightedModule
import numpy as np
import torch.nn.functional as F
from mground.gpu_utils import anpai

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(
            torch.tensor(0.0, device=DEVICE),
            torch.tensor(1.0, device=DEVICE),
        )

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (
            -np.log(np.sqrt(2 * np.pi))
            - torch.log(self.sigma)
            - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)
        ).sum()


# 先验


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, np.exp(sigma1))
        self.gaussian2 = torch.distributions.Normal(0, np.exp(sigma2))

    def log_prob(self, inputs):
        # inputs = inputs.cpu()
        prob1 = torch.exp(self.gaussian1.log_prob(inputs))
        prob2 = torch.exp(self.gaussian2.log_prob(inputs))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(WeightedModule):
    def __init__(self, in_features, out_features, param):
        # 继承父类属性
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重w的参数
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4)
        )
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias 的参数
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).uniform_(-0.2, 0.2)
        )
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(-5, -4)
        )
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # 先验分布
        self.weight_prior = ScaleMixtureGaussian(
            param.pi, param.sigma1, param.sigma2
        )
        self.bias_prior = ScaleMixtureGaussian(
            param.pi, param.sigma1, param.sigma2
        )
        self.log_prior = 0
        self.log_variational_posterior = 0

        self.has_init = True

    def forward(self, input, sample=False, calculate_log_probs=False):

        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(
                weight
            ) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight
            ) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNetwork(WeightedModule):
    # 两层隐藏层
    def __init__(self, param):
        # torch.nn.Module.
        super().__init__()
        self.l1 = BayesianLinear(28 * 28, 400, param)
        self.l2 = BayesianLinear(400, 400, param)
        self.l3 = BayesianLinear(400, 10, param)
        self.param = param
        self.has_init = True

        self.outputs = torch.zeros(5, param.batch_size, param.class_num)

    def forward(self, x, sample=False):

        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return (
            self.l1.log_variational_posterior
            + self.l2.log_variational_posterior
            + self.l2.log_variational_posterior
        )

    def sample_elbo(self, input, target):
        p = self.param

        outputs = torch.zeros(p.samples, p.batch_size, p.class_num)
        log_priors = torch.zeros(p.samples)
        log_var_posteriors = torch.zeros(p.samples)

        outputs, log_priors, log_var_posteriors = anpai(
            [outputs, log_priors, log_var_posteriors], p.use_gpu, False
        )

        for i in range(p.samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_var_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_var_posteriors.mean()

        negative_log_likelihood = F.nll_loss(
            outputs.mean(0), target, size_average=False
        )

        loss = (
            log_variational_posterior - log_prior
        ) / p.class_num + negative_log_likelihood

        return (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
        )

