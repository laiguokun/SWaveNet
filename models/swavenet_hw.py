import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import *;
import math
import random

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

class WaveNetGate(nn.Module):
    
    def __init__(self, input_dim, output_dim, dilate, batch_norm = False, residual = True):
        super(WaveNetGate, self).__init__()
        
        self.filter_conv = nn.Conv1d(input_dim, output_dim, 2, dilation = dilate)
        self.gate_conv = nn.Conv1d(input_dim, output_dim, 2, dilation = dilate)
        self.residual_link = residual;
        if self.residual_link:
            self.residual = nn.Conv1d(input_dim, output_dim, 1)
        
        self.batch_norm = batch_norm;
        if (batch_norm):
            self.filter_conv_bn = nn.BatchNorm1d(output_dim)
            self.gate_conv_bn = nn.BatchNorm1d(output_dim)
            if self.residual_link:
                self.residual_bn = nn.BatchNorm1d(output_dim)
        
    
    def forward(self, inputs):
        
        conv_x, x = inputs
        tanh_x = self.filter_conv(conv_x);
        sigmoid_x = self.gate_conv(conv_x);
        if (self.residual_link):
            residual_x = self.residual(x);
        if self.batch_norm:
            sigmoid_x = self.gate_conv_bn(sigmoid_x);  
            tanh_x = self.filter_conv_bn(tanh_x); 
            if self.residual_link:
                residual_x = self.residual_bn(residual_x);
        sigmoid_x = F.sigmoid(sigmoid_x);
        tanh_x = F.tanh(tanh_x);
        x = tanh_x * sigmoid_x;
        if (self.residual_link):
            return x + residual_x;
        else:
            return x;
        
        
class Gates(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_dim, batch_norm = False):
        super(Gates, self).__init__()
        self.input_dim = input_dim;
        self.output_dim = output_dim;
        self.mlp_dim = mlp_dim;
        self.fc2 = nn.Conv1d(input_dim, output_dim, 1);
        self.batch_norm = batch_norm;
        if batch_norm:
            self.fc2_bn = nn.BatchNorm1d(output_dim) 
        
    def forward(self, x):
        x = self.fc2(F.relu(x));
        if (self.batch_norm):
            x = self.fc2_bn(x);
        return x;
    
class Model(nn.Module):
    def __init__(self, input_dim, embed_dim, z_dim, data):
        super(Model, self).__init__()
        self.input_dim = input_dim;
        self.embed_dim = embed_dim;
        self.output_dim = embed_dim;
        output_dim = embed_dim;
        self.mlp_dim = z_dim;
        mlp_dim = z_dim;
        self.num_layer = 1;
        self.pre_layer = 5;
        self.z_dim = z_dim//self.num_layer;
        self.embedding = nn.Conv1d(input_dim, embed_dim, 1);
        self.batch_norm = False;
        self.data = data;
        
        self.fwds = nn.ModuleList();
        self.skip_gates = nn.ModuleList();
        self.forward_gates = nn.ModuleList();
        self.inference_gates = nn.ModuleList();
        self.backward = nn.ModuleList();
        self.bwd_gates = nn.ModuleList();
        self.pri_gates = nn.ModuleList();
        
        dilate = 1;
        self.fwds_first = nn.ModuleList();
        self.skip0 = Gates(embed_dim, output_dim, mlp_dim)
        self.skip_first = nn.ModuleList();
        for i in range(self.pre_layer):
            self.fwds_first.append(WaveNetGate(embed_dim, embed_dim, dilate, batch_norm = True))
            self.skip_first.append(Gates(embed_dim, output_dim, mlp_dim));
            dilate *= 2;
        
        for i in range(self.num_layer):
            self.fwds.append(WaveNetGate(embed_dim, embed_dim, dilate, batch_norm = True));
            self.backward.append(WaveNetGate(embed_dim, embed_dim, dilate, batch_norm = True));
            
            self.pri_gates.append(Gates(embed_dim, z_dim * 2, mlp_dim));
            self.forward_gates.append(Gates(embed_dim+z_dim, embed_dim, mlp_dim));
            self.inference_gates.append(Gates(embed_dim * 2, z_dim * 2, mlp_dim))
            self.skip_gates.append(Gates(embed_dim+z_dim, output_dim, mlp_dim));
            self.bwd_gates.append(Gates(embed_dim * 2, embed_dim, mlp_dim));
            
            dilate *= 2;
            
        self.final_dilate = dilate // 2;
        
        self.final1 = nn.Conv1d(output_dim, output_dim, 1);
        self.final2 = nn.Conv1d(output_dim, output_dim, 1);
        if self.batch_norm:
            self.final1_bn = nn.BatchNorm1d(output_dim)
            self.final2_bn = nn.BatchNorm1d(output_dim)
        self.k = 1;
        self.mean = nn.Conv1d(output_dim, self.k * 2, 1);
        self.var = nn.Conv1d(output_dim, self.k * 2, 1);
        self.corr = nn.Conv1d(output_dim, self.k, 1);
        if (self.k > 1):
            self.coeff = nn.Conv1d(output_dim, self.k, 1);
        self.binary = nn.Conv1d(output_dim, 1, 1);
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    
    def backward_pass(self, inputs):
        x = inputs;
        x = x.permute(1,2,0);
        x = self.embedding(x);
        
        target = x;
        backwards = []
        dilate = self.final_dilate;
        for i in range(self.num_layer):
            
            conv_x = x.unsqueeze(-1);
            conv_x = F.pad(conv_x, (0, 0, 0, dilate));
            conv_x = conv_x.squeeze(-1);
            inputs = [conv_x, x];
            conv_x = self.backward[self.num_layer - 1 - i](inputs);
            x = conv_x;
            backward_vec = torch.cat([conv_x, target], 1);
            
            backward_vec = self.bwd_gates[i](backward_vec);
            backwards.append(backward_vec);
            
            if (dilate > 1):
                dilate = dilate // 2;
            
        return backwards;
    
    def BiGaussian(self, final, y):
        '''
        mean = self.mean(final);
        logvar = self.var(final);
        #corr = F.tanh(self.corr(final)).squeeze(1);
        binary = F.sigmoid(self.binary(final)).squeeze(1);
        
        y = y.permute(1,2,0);
        y0 = y[:, 0, :]; y1 = y[:, 1, :]; y2 = y[:, 2, :];
        mu_1 = mean[:, 0, :]; mu_2 = mean[:, 1, :];
        logsig_1 = logvar[:, 0, :]; logsig_2 = logvar[:, 1, :];
        sig_1 = torch.exp(logsig_1); sig_2 = torch.exp(logsig_2)
        c_b = y0 * torch.log(binary) + (1 - y0) * torch.log(1 - binary);
        inner1 = (logsig_1 + logsig_2 + math.log(2 * math.pi));
        z = (((y1 - mu_1) / sig_1)**2 + ((y2 - mu_2) / sig_2)**2)
        inner2 = 0.5;
        
        cost = -(inner1 + (inner2 * z));
        ll = (cost + c_b).permute(1,0);
        return ll; 
        '''
        mean = self.mean(final);
        logvar = self.var(final);
        corr = F.tanh(self.corr(final)).squeeze(1);
        binary = F.sigmoid(self.binary(final)).squeeze(1);
        
        y = y.permute(1,2,0);
        y0 = y[:, 0, :]; y1 = y[:, 1, :]; y2 = y[:, 2, :];
        mu_1 = mean[:, 0, :]; mu_2 = mean[:, 1, :];
        logsig_1 = logvar[:, 0, :]; logsig_2 = logvar[:, 1, :];
        sig_1 = torch.exp(logsig_1); sig_2 = torch.exp(logsig_2)
        #c_b = F.binary_cross_entropy(binary, y0, reduce = False)
        c_b = y0 * torch.log(binary) + (1 - y0) * torch.log(1 - binary);
        inner1 = (0.5 * torch.log(1 - corr**2) + 
                  logsig_1 + logsig_2 + math.log(2 * math.pi));
        
        z = (((y1 - mu_1) / sig_1)**2 + ((y2 - mu_2) / sig_2)**2 -
            (2. * (corr * (y1 - mu_1) * (y2 - mu_2)) / (sig_1 * sig_2)))
        
        inner2 = 0.5 * (1./(1. - corr ** 2));
        
        cost = -(inner1 + (inner2 * z));
        ll = (cost + c_b).permute(1,0);
        return ll;  
    
    def BiGMM(self, final, y):
        mean = self.mean(final);
        logvar = self.var(final);
        logvar = torch.clamp(logvar, -8.,8.);
        binary = F.sigmoid(self.binary(final)).squeeze(1);
        coeff = F.log_softmax(self.coeff(final), dim = 1);
        y = y.permute(1,2,0);
        y0 = y[:, 0, :]; y1 = y[:, 1, :].contiguous(); y2 = y[:, 2, :].contiguous();
        mean = mean.view(mean.size(0), 2, self.k, mean.size(-1));
        logvar = logvar.view(logvar.size(0), 2, self.k, logvar.size(-1));
        mu_1 = mean[:, 0, :, :]; mu_2 = mean[:, 1, :, :];
        logsig_1 = logvar[:, 0, :, :]; logsig_2 = logvar[:, 1, :, :];
        sig_1 = torch.exp(logsig_1); sig_2 = torch.exp(logsig_2);
        #c_b = F.binary_cross_entropy(binary, y0, reduce = False)
        c_b = y0 * torch.log(binary) + (1 - y0) * torch.log(1 - binary);
        y1 = y1.view(y1.size(0), 1, y1.size(1)); y2 = y2.view(y2.size(0), 1, y2.size(1));
        inner1 = (logsig_1 + logsig_2 + math.log(2 * math.pi));   
        diff1 = (y1 - mu_1); diff2 = (y2 - mu_2);
        z = ((diff1 / sig_1)**2 + (diff2 / sig_2)**2)  
        
        inner2 = 0.5 ;
        
        cost = -(inner1 + (inner2 * z));
        cost = logsumexp(coeff + cost, dim=1);
        ll = (cost + c_b).permute(1,0);
        return ll;
    
    def forward(self, inputs):
        
        x, y, mask = inputs;
        
        backward_vecs = self.backward_pass(y);
        
        x = x.permute(1,2,0);
        x = F.relu(self.embedding(x));
        
        #first convolution
        dilate = 1;
        final = self.skip0(x);
        for i in range(self.pre_layer):
            conv_x = x.unsqueeze(-1);
            conv_x = F.pad(conv_x, (0, 0, dilate, 0));
            conv_x = conv_x.squeeze(-1);
            inputs = [conv_x, x];
            next_x = self.fwds_first[i](inputs);
            final = final + self.skip_first[i](next_x);
            x = next_x
            dilate *=2;
        
        # forward
        kld_loss = 0;
        for i in range(self.num_layer):
            conv_x = x.unsqueeze(-1);
            conv_x = F.pad(conv_x, (0, 0, dilate, 0));
            conv_x = conv_x.squeeze(-1);
            inputs = [conv_x, x];
            next_x = self.fwds[i](inputs);
            
            z_pri = self.pri_gates[i](next_x);
            z_pri = torch.clamp(z_pri, -8.,8.);
            z_mu, z_theta = torch.chunk(z_pri, 2, 1);
            
            z_post2 = backward_vecs[self.num_layer - 1 - i];
            z_post = torch.cat([next_x, z_post2], 1);
            z_post = self.inference_gates[i](z_post);
            z_post = torch.clamp(z_post, max = 8.);
            mu, theta = torch.chunk(z_post, 2, 1);
            
            z = self.reparameterize(mu, theta);
            
            #compute KL(q||p)
            tmp = gaussian_kld([mu, theta], [z_mu, z_theta]);
            tmp = tmp.permute(2,0,1);
            tmp = (tmp.sum(-1) * mask).sum(0);
            tmp = tmp.mean();
            kld_loss += tmp;
            
            tmp = torch.cat([next_x, z], 1);
            tmp = self.skip_gates[i](tmp);
            final = final + tmp;
            next_x = torch.cat([next_x, z], 1);
            x = self.forward_gates[i](next_x);
            if (i >= 0):
                dilate *= 2;
            
            
        final = self.final1(F.relu(final));
        if (self.batch_norm):
            final = self.final1_bn(final);
        final = self.final2(F.relu(final));
        if (self.batch_norm):
            final = self.final2_bn(final);
        #final = final.permute(2,0,1);
        if (self.k > 1):
            loss = self.BiGMM(final, y) 
        else:
            loss = self.BiGaussian(final, y);
        loss = (loss * mask).sum(0);
        loss = loss.mean();
        return -loss, -kld_loss
    
    def sample(self, final):
        
        mean = self.mean(final);
        logvar = self.var(final);
        binary = F.sigmoid(self.binary(final)).squeeze(1);
        
        mu = mean[:,:,-1].cpu().data.squeeze(); 
        logvar = logvar[:,:,-1].cpu().data.squeeze();
        b = binary[:,-1].cpu().data.squeeze();
        std = logvar.exp_()
        eps = std.new(std.size()).normal_()
        x = eps.mul(std).add_(mu)
        
        r = torch.rand(b.size());
        b = (r < b).float();
        return x, b;
    
    def sample_gmm(self, final):
        mean = self.mean(final);
        logvar = self.var(final);
        binary = F.sigmoid(self.binary(final)).squeeze(1);
        coeff = F.softmax(self.coeff(final), dim = 1);
        coeff = coeff[:,:,-1].cpu().data.squeeze();

        mean = mean[:,:,-1].cpu().data.squeeze(); 
        logvar = logvar[:,:,-1].cpu().data.squeeze();
        mean = mean.view(2, self.k);
        logvar = logvar.view(2, self.k);
        r = random.random();
        for i in range(self.k):
            r = r - coeff[i];
            if (r <= 0):
                mean = mean[:,i];
                logvar = logvar[:,i];
                std = logvar.exp_()
                eps = std.new(std.size()).normal_()
                x = eps.mul(std).add_(mean)       
                break;
        b = binary[:,-1].cpu().data.squeeze();
        r = torch.rand(b.size());
        b = (r < b).float();
        return x, b;
        
    def generate(self, inputs):
        
        #x, y = inputs;
        #backward_vecs = self.backward_pass(y);
        
        x, eps = inputs
        x = x.permute(1,2,0);
        x = F.relu(self.embedding(x));
        dilate = 1;
        final = self.skip0(x);
        for i in range(self.pre_layer):
            conv_x = x.unsqueeze(-1);
            conv_x = F.pad(conv_x, (0, 0, dilate, 0));
            conv_x = conv_x.squeeze(-1);
            inputs = [conv_x, x];
            next_x = self.fwds_first[i](inputs);
            final = final + self.skip_first[i](next_x);
            x = next_x
            dilate *=2;
            
        for i in range(self.num_layer):
            conv_x = x.unsqueeze(-1);
            conv_x = F.pad(conv_x, (0, 0, dilate, 0));
            conv_x = conv_x.squeeze(-1);
            inputs = [conv_x, x];
            next_x = self.fwds[i](inputs);
            
            z_pri = self.pri_gates[i](next_x);
            z_pri = torch.clamp(z_pri, -8.,8.);
            z_mu, z_theta = torch.chunk(z_pri, 2, 1);
            
            z = z_mu
            
            tmp = torch.cat([next_x, z], 1);
            tmp = self.skip_gates[i](tmp);
            final = final + tmp;
            next_x = torch.cat([next_x, z], 1);
            x = self.forward_gates[i](next_x);
            if (i >= 0):
                dilate *= 2;
            
            
        final = self.final1(F.relu(final));
        if (self.batch_norm):
            final = self.final1_bn(final);
        final = self.final2(F.relu(final));
        if (self.batch_norm):
            final = self.final2_bn(final);
        if (self.k == 1):
            x = self.sample(final)
        else:
            x = self.sample_gmm(final);
        return x