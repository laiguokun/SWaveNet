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
from blizzard_data import Blizzard_tbptt

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
    return end_kd + (init_kd - end_kd) * ((math.cos(0.5 * math.pi * float(epoch) / total_epoch)));

def evaluate(dataset, model, args, split='valid'):
    model.eval()
    loss_sum = 0
    cnt = 0;
    length = 40
    #print(dataset)
    for x, y, x_mask in dataset:
        #print(x, y, x_mask)
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
    return -loss_sum/cnt;


parser = argparse.ArgumentParser(description='PyTorch VAE for sequence')
parser.add_argument('--expname', type=str, default='tiny')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--data', type=str, default='/usr1/glai1/datasets/')
parser.add_argument('--file_name', type=str, default='blizzard_tbptt')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--end_lr', type=float, default=0.)
parser.add_argument('--kld', type=str, default='True')
parser.add_argument('--model_name', type=str, default='swavenet')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--embed_size', type=int, default=1024)
parser.add_argument('--z_size', type=int, default=512)
parser.add_argument('--resume', type=str, default=None)


args = parser.parse_args()

print(args);

seed = args.seed; expname = args.expname; num_epochs = args.num_epochs; data = args.data; lr = args.lr; 
model_name = args.model_name;batch_size = args.batch_size;

torch.cuda.set_device(args.gpu)
rng = np.random.RandomState(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed);
log_interval = 500
model_id = 'blizzard_seed{}'.format(seed)
if not os.path.exists(expname):
    os.makedirs(expname)
log_file_name = os.path.join(expname, model_id + '.txt')
model_file_name = os.path.join(expname, model_id + '.pt')
log_file = open(log_file_name, 'w')

print('Loading data..')
#X_mean = -0.35828044227
#X_std = 3117.59272379
bsz = 128

#file_name = 'blizzard_unseg_tbptt'
normal_path = "os.path.join(data, args.file_name + '_normal.npz')"
normal_params = np.load(os.path.join(data, args.file_name + '_normal.npz'))
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']



train_data = Blizzard_tbptt(name='train',
                            path=args.data,
                            frame_size=200,
                            file_name=args.file_name,
                            X_mean=X_mean,
                            X_std=X_std)
valid_data = Blizzard_tbptt(name='valid',
                            path=args.data,
                            frame_size=200,
                            file_name=args.file_name,
                            X_mean=X_mean,
                            X_std=X_std)
test_data = Blizzard_tbptt(name='test',
                           path=args.data,
                           frame_size=200,
                           file_name=args.file_name,
                           X_mean=X_mean,
                           X_std=X_std)

print('X_mean: %f, X_std: %f', (X_mean, X_std))

assert bsz == 128
train_data = load.BlizzardIterator(train_data, bsz, start=0, end=2040064)
valid_data = load.BlizzardIterator(valid_data, bsz, start=2040064, end=2152704)
# Use complete batch only.
test_data = load.BlizzardIterator(test_data, bsz, start=2152704, end=2267008-128)
print('Done.')
# Build model
nbatches = train_data.nbatch
total_step = num_epochs * nbatches;
if args.model_name[:10] == 'wavenetvae':
    model = eval(args.model_name).Model(input_dim=200, embed_dim=args.embed_size, z_dim=args.z_size, data=None)
else:
    model = eval(args.model_name).Model(input_dim=200, embed_dim=512, output_dim=1024, data=None)
model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
kld_step = 0.00005
if (args.resume == None):
    step = 0
    kld_weight = kld_step;
    start_epoch = 0;
else:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        step = (start_epoch) * nbatches;
        kld_weight = adjust_kd(step, 20 * nbatches, kld_step, 1.);
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))    
        
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)
t = timeit.default_timer()

loss_sum = 0;
kld_loss_sum = 0;
logp_loss_sum = 0;
flag = True;
#print('test', evaluate(test_data, model, args, split='test'))
for epoch in range(start_epoch, num_epochs):
    old_valid_loss = np.inf
    model.train()
    print('Epoch {}: ({})'.format(epoch, model_id.upper()))
    for x, y, x_mask in train_data:
        opt.zero_grad()
        x = Variable(torch.from_numpy(x)).float().cuda()
        y = Variable(torch.from_numpy(y)).float().cuda()
        x_mask = Variable(torch.from_numpy(x_mask)).float().cuda()
        if (args.kld == 'True'):
            loss, kld_loss = model([x,y,x_mask]);
            total_loss = loss - kld_loss * kld_weight;
            if np.isnan(total_loss.data[0]) or np.isinf(total_loss.data[0]):
                print("NaN")  # Useful to see if training is stuck.
                flag = False;
                break
            total_loss.backward();
            total_loss = total_loss.data[0];
            kld_loss_sum += kld_loss.data[0];
            logp_loss_sum += loss.data[0];
        else:
            all_loss = model([x,y,x_mask]);
            if np.isnan(all_loss.data[0]) or np.isinf(all_loss.data[0]):
                continue
            all_loss.backward()
            total_loss = all_loss.data[0]
            
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.1, 'inf')
        opt.step()
        loss_sum += total_loss;
        step += 1;
        if step % log_interval == 0:
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
            
        kld_weight = adjust_kd(step, 20 * nbatches, kld_step, 1.);
        adjust_lr(opt, step, total_step, args.lr, args.end_lr);
    
    if flag == False:
        break;

    print(float(step)/total_step);
    # evaluate per epoch
    print('--- Epoch finished ----')

    val_loss = evaluate(valid_data, model, args)
    log_line = 'validation  -- epoch: %s, nll: %f' % (epoch, val_loss)
    print(log_line)
    log_file.write(log_line + '\n')
    
    test_loss = evaluate(test_data, model, args, split='test')
    log_line = 'test -- epoch: %s, nll: %f' % (epoch, test_loss)
    print(log_line)
    log_file.write(log_line + '\n')
    log_file.flush()
    
