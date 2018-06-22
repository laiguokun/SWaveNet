import torch
import torch.nn as nn
from torch.autograd import Variable
import timeit
import argparse
import numpy as np
import os
import random
import load
from models import swavenet_hw
import math;
from iamondb import IAMOnDB

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    
def adjust_lr(optimizer, epoch, total_epoch, init_lr, end_lr):
    assert init_lr > end_lr;
    lr = end_lr + (init_lr - end_lr) * (0.5 * (1+math.cos(math.pi * float(epoch) / total_epoch))); 
    #print(lr);
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def adjust_kd(epoch, total_epoch, init_kd, end_kd):
    if (epoch > total_epoch):
        return 1.;
    return end_kd + (init_kd - end_kd) * (0.5 * (1 + math.cos(math.pi * float(epoch) / total_epoch)));

def evaluate(dataset, model, args, split='valid'):
    model.eval()
    loss_sum = 0
    cnt = 0;
    for x, y, x_mask in dataset:
        try:
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
        except:
            print('exception');
    return -loss_sum/cnt;


parser = argparse.ArgumentParser(description='PyTorch VAE for sequence')
parser.add_argument('--expname', type=str, default='tiny')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--data', type=str, default='/usr0/home/glai1/dataset/iamondb')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--end_lr', type=float, default=0.)
parser.add_argument('--kld', type=str, default='True')
parser.add_argument('--model_name', type=str, default='swavenet_hw')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--embed_size', type=int, default=200)
parser.add_argument('--z_size', type=int, default=200)


args = parser.parse_args()

print(args);

seed = args.seed; expname = 'tiny'; num_epochs = args.num_epochs; data = args.data; lr = args.lr; 
model_name = args.model_name;batch_size = args.batch_size;

torch.cuda.set_device(args.gpu)
rng = np.random.RandomState(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed);
model_id = 'IAMOnDB_seed{}'.format(seed)
if not os.path.exists(expname):
    os.makedirs(expname)
log_file_name = os.path.join(expname, model_id + '.txt')
model_file_name = os.path.join(expname, model_id + '.pt')
log_file = open(log_file_name, 'w')
data_path = args.data
print('Loading data..')
bsz = args.batch_size

train_data = IAMOnDB(name='train',
                     prep='normalize',
                     cond=False,
                     path=data_path)

X_mean = train_data.X_mean
X_std = train_data.X_std

valid_data = IAMOnDB(name='valid',
                     prep='normalize',
                     cond=False,
                     path=data_path,
                     X_mean=X_mean,
                     X_std=X_std)

print('X_mean: %f, X_std: %f', (X_mean, X_std))

train_data = load.IAMOnDBIterator(train_data, bsz)
valid_data = load.IAMOnDBIterator(valid_data, bsz)
print('Done.')
# Build model
if args.model_name[:10] == 'wavenetvae':
    model = eval(args.model_name).Model(input_dim=3, embed_dim=args.embed_size, z_dim=args.z_size, data=None)
else:
    model = eval(args.model_name).Model(input_dim=3, embed_dim=256, output_dim=256, data=None)
model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-5)

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)
nbatches = train_data.nbatch
t = timeit.default_timer()
kld_step = 0.00005
kld_weight = kld_step;

step = 0
total_step = num_epochs * nbatches;
loss_sum = 0;
kld_loss_sum = 0;
logp_loss_sum = 0;
log_interval = nbatches;
for epoch in range(num_epochs):
    old_valid_loss = np.inf
    model.train()
    print('Epoch {}: ({})'.format(epoch, model_id.upper()))
    for x, y, x_mask in train_data:
        try:
            opt.zero_grad()
            x = Variable(torch.from_numpy(x)).float().cuda()
            y = Variable(torch.from_numpy(y)).float().cuda()
            x_mask = Variable(torch.from_numpy(x_mask)).float().cuda()
            if (args.kld == 'True'):
                loss, kld_loss = model([x,y,x_mask]);
                total_loss = loss - kld_loss * kld_weight;
                if np.isnan(total_loss.data[0]) or np.isinf(total_loss.data[0]):
                    print("NaN")  # Useful to see if training is stuck.
                    continue
                total_loss.backward();
                total_loss = total_loss.data[0];
                kld_loss_sum += kld_loss.data[0];
                logp_loss_sum += loss.data[0];
            else:
                all_loss = model([x,y,x_mask]);
                if np.isnan(all_loss.data[0]) or np.isinf(all_loss.data[0]):
                    print('NaN');
                    continue
                all_loss.backward()
                total_loss = all_loss.data[0]

            torch.nn.utils.clip_grad_norm(model.parameters(), 0.1, 'inf')
            opt.step()
            loss_sum += total_loss;
            step += 1;
        except:
            print('exception')
        
    s = timeit.default_timer()
    log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f, logp_loss:%f, kld_loss: %f, actual_loss:%f' % (
        s-t, epoch, num_epochs, step % nbatches, nbatches,
        -loss_sum / log_interval, -logp_loss_sum/log_interval, -kld_loss_sum/log_interval, -(logp_loss_sum-kld_loss_sum)/log_interval)
    print(log_line)
    log_file.write(log_line + '\n')
    log_file.flush()
    loss_sum = 0;
    kld_loss_sum = 0;
    logp_loss_sum = 0;
    
    kld_weight = adjust_kd(step, int(0.5 * num_epochs * nbatches), kld_step, 1.);
    adjust_lr(opt, step, total_step, args.lr, args.end_lr);
    
    print(float(step)/total_step, kld_weight);
    # evaluate per epoch
    print('--- Epoch finished ----')

    val_loss = evaluate(valid_data, model, args)
    log_line = 'validation  -- epoch: %s, nll: %f' % (epoch, val_loss)
    print(log_line)
    log_file.write(log_line + '\n')
    if ((epoch+1) % 10 == 0):
        log_file_name = 'iamondb_models/{}-{}-{}-{}-epoch-{}-{}.tar'.format(args.model_name, args.embed_size, args.z_size, args.lr, epoch + 1, args.expname);
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model_name,
            'state_dict': model.state_dict(),
            'optimizer' : opt.state_dict(),
        }, filename  = log_file_name)
