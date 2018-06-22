import torch
import torch.nn as nn
import torch.nn.functional as F
from Regressor import Regressor
from utils import *;


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
        #print(conv_x.size(), x.size());
        tanh_x = self.filter_conv(conv_x);
        sigmoid_x = self.gate_conv(conv_x);
        if (self.residual_link):
            residual_x = self.residual(x);
        if self.batch_norm:
            sigmoid_x = self.gate_conv_bn(sigmoid_x);  
            tanh_x = self.filter_conv_bn(tanh_x); 
            if self.residual_link:
                residual_x = self.residual_bn(residual_x);
        sigomid_x = F.sigmoid(sigmoid_x);
        tanh_x = F.tanh(tanh_x);
        x = tanh_x * sigmoid_x;
        #print(x.size(), residual_x.size());
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
        self.fc1 = nn.Conv1d(input_dim, mlp_dim, 1)
        self.fc2 = nn.Conv1d(mlp_dim, output_dim, 1);
        self.batch_norm = batch_norm;
        if batch_norm:
            self.fc1_bn = nn.BatchNorm1d(mlp_dim)
            self.fc2_bn = nn.BatchNorm1d(output_dim) 
        
    def forward(self, x):
        x = self.fc1(x);
        if (self.batch_norm):
            x = self.fc1_bn(x);
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
        self.num_layer = 4;
        self.pre_layer = 0;
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
        
        #self.bwd_fc1 = nn.Conv1d(embed_dim, output_dim, 1);
        #self.bwd_fc2 = nn.Conv1d(output_dim, output_dim, 1);
        
        if self.batch_norm:
            self.final1_bn = nn.BatchNorm1d(output_dim)
            self.final2_bn = nn.BatchNorm1d(output_dim)
        self.loss = 'Gaussian';
        self.regressor = Regressor(self.loss, output_dim, input_dim);
        self.dropout = nn.Dropout(0.);
    
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
            z_post = torch.clamp(z_post, max=8.);
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
        final = final.permute(2,0,1);
        outputs = self.regressor(final);
        loss = LogLikelihood(y, outputs, self.loss, self.data);
        loss = loss.sum(-1);
        loss = (loss * mask).sum(0);
        loss = loss.mean();
        return -loss, -kld_loss
    