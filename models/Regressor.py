import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self, Loss, hid, m):
        super(Regressor, self).__init__()
        self.loss_function = Loss;
        self.m = m
        if (self.loss_function == 'uni-Gaussian-novar'):
            self.mean = nn.Linear(hid, self.m);
            self.var = torch.rand(1) * -1;
            self.var = nn.Parameter(self.var.cuda());
        
        
        if (self.loss_function == 'Gaussian'):
            
            self.mean = nn.Linear(hid, self.m);
            self.var = nn.Linear(hid, self.m);
            
        if (self.loss_function[:12] == 'mul-Gaussian'):
            
            self.K = int(self.loss_function.split('@')[-1]);
            self.mean = nn.Linear(hid, self.m * self.K);
            self.var = nn.Linear(hid, self.m * self.K);
            self.pi = nn.Linear(hid, self.m * self.K);
        
        if (self.loss_function[:7] == 'softmax'):
            
            self.K = int(self.loss_function.split('@')[-1]);
            self.softmax = nn.Linear(hid, self.m * self.K)
            
    def forward(self, X):
        
        batch_size, m = X.size(0), self.m;
        
        if (self.loss_function == 'uni-Gaussian-novar'):
            mean = self.mean(X);
            var = self.var.view(1,1).expand(batch_size, m).contiguous();
            return [mean, var];
        
        if (self.loss_function == 'Gaussian'):
            
            return [self.mean(X), self.var(X)];
            
        if (self.loss_function[:12] == 'mul-Gaussian'):
            
            seqlen, batch_size, _ = X.size();
            mean = self.mean(X).view(seqlen, batch_size, m, self.K); 
            var = self.var(X).view(seqlen, batch_size, m, self.K);
            pi = self.pi(X).view(seqlen * batch_size * m, self.K);
            pi = F.softmax(pi);
            pi = pi.view(seqlen, batch_size, m, self.K);
            return [mean, var, pi];
            
        
        if (self.loss_function[:7] == 'softmax'):
            seqlen, batch_size, _ = X.size();
            predict = self.softmax(X).view(seqlen * batch_size * m, self.K);
            predict = F.log_softmax(predict);
            predict = predict.view(seqlen, batch_size, m, self.K)
            return [predict];
        
        print('no loss found');
