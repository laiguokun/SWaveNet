import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Regressor import Regressor
from models.utils import LogLikelihood


class Model(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, data):
        super(Model, self).__init__()
        self.input_dim = input_dim;
        self.embed_dim = embed_dim;
        self.output_dim = output_dim;
        self.num_layer = 5;
        self.embedding = nn.Conv1d(input_dim, embed_dim, 1);
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList();
        self.residuals = nn.ModuleList();
        #self.residuals_bns = nn.ModuleList()
        self.skip = nn.ModuleList();
        self.skip.append(nn.Conv1d(embed_dim, embed_dim, 1));
        self.batch_norm = False;
        self.data = data;
        
        if self.batch_norm:
            self.filter_conv_bns = nn.ModuleList()
            self.gate_conv_bns = nn.ModuleList();
            self.residual_bns = nn.ModuleList();
            self.skip_bns = nn.ModuleList();
            self.skip_bns.append(nn.BatchNorm1d(embed_dim));
        
        dilate = 1;
        self.kernel_size = 2;
        for i in range(self.num_layer):
            self.filter_convs.append(nn.Conv1d(embed_dim, embed_dim, self.kernel_size, dilation = dilate));
            self.gate_convs.append(nn.Conv1d(embed_dim, embed_dim, self.kernel_size, dilation = dilate));
            self.residuals.append(nn.Conv1d(embed_dim, embed_dim, 1));
            self.skip.append(nn.Conv1d(embed_dim, embed_dim, 1));
            
            if self.batch_norm:
                self.filter_conv_bns.append(nn.BatchNorm1d(embed_dim))
                self.gate_conv_bns.append(nn.BatchNorm1d(embed_dim))
                self.residual_bns.append(nn.BatchNorm1d(embed_dim))
                self.skip_bns.append(nn.BatchNorm1d(embed_dim));
                
            dilate *= 2;
        
        self.final1 = nn.Conv1d(embed_dim, embed_dim, 1);
        self.final2 = nn.Conv1d(embed_dim, embed_dim, 1);
        
        if self.batch_norm:
            self.final1_bn = nn.BatchNorm1d(embed_dim)
            self.final2_bn = nn.BatchNorm1d(embed_dim)
        #self.loss = 'Gaussian';
        self.loss = 'mul-Gaussian@20';
        self.regressor = Regressor(self.loss, embed_dim, input_dim);
        self.dropout = nn.Dropout(0.);
    
    def forward(self, inputs):
        
        x, y, mask = inputs;
        x = x.permute(1,2,0);
        x = F.relu(self.embedding(x));
        final = self.skip[0](x);
        dilate = 1;
        # wavenet forward
        for i in range(self.num_layer):
            conv_x = x.unsqueeze(-1);
            conv_x = F.pad(conv_x, (0, 0, dilate, 0));
            conv_x = conv_x.squeeze(-1);
            tanh_x = self.filter_convs[i](conv_x);
            sigmoid_x = self.gate_convs[i](conv_x);
            residual_x = self.residuals[i](x);
            
            if self.batch_norm:
                sigmoid_x = self.gate_conv_bns[i](sigmoid_x);  
                tanh_x = self.filter_conv_bns[i](tanh_x); 
                residual_x = self.residual_bns[i](residual_x);
                
            sigmoid_x = F.sigmoid(sigmoid_x);
            tanh_x = F.tanh(tanh_x);
            
            x = tanh_x * sigmoid_x;
            skip_x = self.skip[i+1](x);
            if self.batch_norm:
                skip_x = self.skip_bns[i+1](skip_x);
            x = skip_x + residual_x
            final = skip_x + final;
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
        return -loss
    
