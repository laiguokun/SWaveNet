import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Regressor import Regressor
from models.utils import LogLikelihood


class Model(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, data=None):
        super(Model, self).__init__()
        self.input_dim = input_dim;
        self.embed_dim = embed_dim;
        self.output_dim = output_dim;
        self.embedding1 = nn.Linear(input_dim, embed_dim);
        self.embedding2 = nn.Linear(embed_dim, embed_dim);
        self.rnn = nn.LSTM(embed_dim, output_dim);
        self.loss = 'mul-Gaussian@20'
        self.regressor = Regressor(self.loss, embed_dim, input_dim);
        self.final1 = nn.Linear(output_dim, embed_dim);
        self.final2 = nn.Linear(embed_dim, embed_dim);
        self.dropout = nn.Dropout(0.);
    
    def forward(self, inputs):
        
        x, y, mask = inputs;
        x = F.relu(self.embedding1(x));
        x = F.relu(self.embedding2(x));
        output, hn = self.rnn(x);
        output = self.dropout(output);
        output = F.relu(self.final1(output));
        output = F.relu(self.final2(output));
        outputs = self.regressor(output);
        loss = LogLikelihood(y, outputs, self.loss);
        loss = loss.sum(-1);
        loss = (loss * mask).sum(0);
        loss = loss.mean();
        return -loss
    