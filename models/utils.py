import torch
import torch.nn as nn
import torch.nn.functional as F
import math;
import numpy as np
from torch.autograd import Variable

def gaussian_kld(left, right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    mu_left, logvar_left = left; mu_right, logvar_right = right
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (torch.exp(logvar_left) / torch.exp(logvar_right)) +
                        ((mu_left - mu_right)**2.0 / torch.exp(logvar_right)) - 1.0)
    return gauss_klds

def LogLikelihood(target, inputs, loss, data=None):
    
    if (loss == 'Gaussian'):
        mean = inputs[0]; logvar = inputs[1];
        logvar = torch.clamp(logvar, -12., 12.)
        var = torch.exp(logvar);
        diff = target - mean;
        res = -torch.pow(diff,2)/(2 * torch.pow(var, 2));
        res = -0.5 * math.log(2 * math.pi) - logvar + res 
        return res;
    
    if (loss[:12] == 'mul-Gaussian'):
        mean = inputs[0]; logvar = inputs[1]; pi = inputs[2];
        logvar = torch.clamp(logvar, -8., 8.)
        var = torch.exp(logvar);
        target = target.unsqueeze(-1);
        target = target.expand(mean.size());
        diff_ = target - mean;
        diff = -torch.pow(diff_,2)/(2 * torch.pow(var, 2))
        max_diff = diff.max(-1,keepdim=True)[0];
        tmp = max_diff.expand(diff.size())
        diff = torch.exp(diff - tmp) + 1e-12;
        res = (1. / var) * diff * pi;
        res = res.sum(-1);
        res = (torch.log(res) + max_diff.squeeze(-1)) - 0.5 * math.log(2 * math.pi);
        return res;
    
    if (loss[:7] == 'softmax'):
        
        X = inputs[0];
        batch_size, seqlen, dim, K  = X.size();
        n_samples = batch_size * seqlen * dim
        X = X.view(-1,);
        target = (target - data.min)/(data.max - data.min);
        #assert target.max().data[0] <=1;
        #assert target.min().data[0] >=0;
        target = torch.clamp(target, 0., 0.9999);
        target = torch.floor(target * K).view(n_samples,);
        tmp = Variable(torch.arange(0,n_samples).cuda() * K);
        tmp = tmp + target;
        tmp = tmp.long();
        res = X[tmp].view(batch_size, seqlen, dim);
        return res;
