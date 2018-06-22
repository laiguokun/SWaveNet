import torch
import torch.nn as nn
from torch.autograd import Variable
import timeit
import argparse
import numpy as np
import os
import random
import load
from models import  rnn, wavenet, swavenet
import math;

def adjust_lr(optimizer, epoch, total_epoch, init_lr, end_lr):
    assert init_lr > end_lr;
    lr = end_lr + (init_lr - end_lr) * (0.5 * (1+math.cos(math.pi * float(epoch) / total_epoch))); 
    print(lr);
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_kd(epoch, total_epoch, init_kd, end_kd):
    if (epoch > total_epoch):
        return 1.;
    return end_kd + (init_kd - end_kd) * ((math.cos(0.5 * math.pi * float(epoch) / total_epoch)));

def evaluate(dataset, model, args, split='valid'):
    def get_batch():
        if split == 'valid':
            return dataset.get_valid_batch()
        else:
            return dataset.get_test_batch()

    model.eval()
    loss_sum = 0
    cnt = 0;
    length = 40
    for x, y, x_mask in get_batch():
        if split == 'valid':
            x = Variable(torch.from_numpy(x), volatile=True).float().cuda()
            y = Variable(torch.from_numpy(y), volatile=True).float().cuda()
            x_mask = Variable(torch.from_numpy(x_mask), volatile=True).float().cuda()
            if (args.kld == 'True'):
                loss, kld_loss = model([x,y,x_mask]);
                total_loss = loss - kld_loss;
                total_loss = total_loss.data[0];
            else:
                all_loss = model([x,y,x_mask]);
                total_loss = all_loss.data[0]
            loss_sum += total_loss;
            cnt += 1;
        else:
            l = 0.
            for i in range(0, x.shape[0], length):
                x_ = Variable(torch.from_numpy(x[i:i+length]), volatile=True).float().cuda()
                y_ = Variable(torch.from_numpy(y[i:i+length]), volatile=True).float().cuda()
                x_mask_ = Variable(torch.from_numpy(x_mask[i:i+length]), volatile=True).float().cuda()
                if (args.kld == 'True'):
                    loss, kld_loss = model([x_,y_,x_mask_]);
                    total_loss = loss - kld_loss;
                    total_loss = total_loss.data[0];
                else:
                    all_loss = model([x_,y_,x_mask_]);
                    total_loss = all_loss.data[0]
                l += total_loss;
            loss_sum += l;
            cnt += 1;
    return -loss_sum/cnt;


parser = argparse.ArgumentParser(description='PyTorch VAE for sequence')
parser.add_argument('--expname', type=str, default='timit_logs')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--data', type=str, default='./data/')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--end_lr', type=float, default=0.)
parser.add_argument('--kld', type=str, default='True')
parser.add_argument('--model_name', type=str, default='swavenet')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--embed_size', type=int, default=1024)
parser.add_argument('--z_size', type=int, default=512)
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

print(args);

seed = args.seed; expname = args.expname; num_epochs = args.num_epochs; data = args.data; lr = args.lr; 
model_name = args.model_name;batch_size = args.batch_size;

torch.cuda.set_device(args.gpu)
rng = np.random.RandomState(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed);
log_interval = 100
model_id = 'timit_seed{}'.format(seed)
if not os.path.exists(expname):
    os.makedirs(expname)
log_file_name = os.path.join(expname, model_id + '.txt')
model_file_name = os.path.join(expname, model_id + '.pt')
log_file = open(log_file_name, 'w')

print('Loading data..')
timit = load.TimitData(data + 'timit_raw_batchsize64_seqlen40.npz', batch_size)
print('Done.')
model = eval(model_name).Model(200, args.embed_size, args.z_size, timit);
model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)
nbatches = timit.u_train.shape[0] // batch_size
t = timeit.default_timer()
kld_step = 0.00005
kld_weight = kld_step;

for epoch in range(num_epochs):
    step = 0
    old_valid_loss = np.inf
    model.train()
    loss_sum = 0;
    kld_loss_sum = 0;
    logp_loss_sum = 0;
    print('Epoch {}: ({})'.format(epoch, model_id.upper()))
    for x, y, x_mask in timit.get_train_batch():
        opt.zero_grad()
        x = Variable(torch.from_numpy(x)).float().cuda()
        y = Variable(torch.from_numpy(y)).float().cuda()
        x_mask = Variable(torch.from_numpy(x_mask)).float().cuda()
        if (args.kld == 'True'):
            loss, kld_loss = model([x,y,x_mask]);
            total_loss = loss - kld_loss * kld_weight;
            if np.isnan(total_loss.data[0]) or np.isinf(total_loss.data[0]):
                print("NaN")  # Useful to see if training is stuck.
                continue;
            total_loss.backward();
            total_loss = total_loss.data[0];
            kld_loss_sum += kld_loss.data[0];
            logp_loss_sum += loss.data[0];
        else:
            all_loss = model([x,y,x_mask]);
            all_loss.backward()
            total_loss = all_loss.data[0]
            
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.1, 'inf')
        opt.step()
        loss_sum += total_loss;
        step += 1;
        if step % log_interval == 0:
            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f, logp_loss:%f, kld_loss: %f' % (
                s-t, epoch, num_epochs, step, nbatches,
                -loss_sum / step, -logp_loss_sum/step, -kld_loss_sum/step)
            print(log_line)
            log_file.write(log_line + '\n')
            log_file.flush()
            
    kld_weight = adjust_kd(epoch, 200, kld_step, 1.);
    adjust_lr(opt, epoch, num_epochs, args.lr, args.end_lr);
    if ((epoch+1) % 10 == 0):
        print('--- Epoch finished ----')
        val_loss = evaluate(timit, model, args)
        log_line = 'valid -- epoch: %s, nll: %f' % (epoch, val_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        test_loss = evaluate(timit, model, args, split='test')
        log_line = 'test -- epoch: %s, nll: %f' % (epoch, test_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        log_file.flush()
test_loss = evaluate(timit, model, args, split='test')
log_line = 'test -- epoch: %s, nll: %f' % (epoch, test_loss)
print(log_line)
log_file.write(log_line + '\n')
log_file.flush()


