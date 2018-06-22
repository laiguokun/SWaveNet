import numpy as np
import numpy.random as npr
from scipy.io import loadmat
import os
import json
from collections import defaultdict, OrderedDict
import math;
import random
import torch;

def chunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    for i in range(0, len(sequence), n):
        yield sequence[i:i + n]


class TimitData():
    def __init__(self, fn, batch_size):
        data = np.load(fn)

        ####
        # IMPORTANT: u_train is the input and x_train is the target.
        ##
        u_train, x_train = data['u_train'], data['x_train']
        u_valid, x_valid = data['u_valid'], data['x_valid']
        (u_test, x_test, mask_test) = data['u_test'],  data['x_test'], data['mask_test']

        # assert u_test.shape[0] == 1680
        # assert x_test.shape[0] == 1680
        # assert mask_test.shape[0] == 1680

        self.u_train = u_train
        self.x_train = x_train
        self.u_valid = u_valid
        self.x_valid = x_valid
        
        self.max = float(max(u_train.max(), u_valid.max()));
        self.min = float(min(u_train.min(), u_valid.min()));
        print(self.max,self.min);
        
        # make multiple of batchsize
        n_test_padded = ((u_test.shape[0] // batch_size) + 1)*batch_size
        assert n_test_padded > u_test.shape[0]
        pad = n_test_padded - u_test.shape[0]
        u_test = np.pad(u_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        x_test = np.pad(x_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        mask_test = np.pad(mask_test, ((0, pad), (0, 0)), mode='constant')
        self.u_test = u_test
        self.x_test = x_test
        self.mask_test = mask_test

        self.n_train = u_train.shape[0]
        self.n_valid = u_valid.shape[0]
        self.n_test = u_test.shape[0]
        self.batch_size = batch_size

        print("TRAINING SAMPLES LOADED", self.u_train.shape)
        print("TEST SAMPLES LOADED", self.u_test.shape)
        print("VALID SAMPLES LOADED", self.u_valid.shape)
        print("TEST AVG LEN        ", np.mean(self.mask_test.sum(axis=1)) * 200)
        # test that x and u are correctly shifted
        assert np.sum(self.u_train[:, 1:] - self.x_train[:, :-1]) == 0.0
        assert np.sum(self.u_valid[:, 1:] - self.x_valid[:, :-1]) == 0.0
        for row in range(self.u_test.shape[0]):
            l = int(self.mask_test[row].sum())
            if l > 0:  # if l is zero the sequence is fully padded.
                assert np.sum(self.u_test[row, 1:l] -
                              self.x_test[row, :l-1]) == 0.0, row

    def _iter_data(self, u, x, mask=None, shuffle=False):
        # u refers to the input whereas x, to the target.
        indices = np.arange(len(u))
        if (shuffle == True):
            np.random.shuffle(indices);
        for idx in chunk(indices, n=self.batch_size):
            u_batch, x_batch = u[idx], x[idx]
            if mask is None:
                mask_batch = np.ones((x_batch.shape[0], x_batch.shape[1]), dtype='float32')
            else:
                mask_batch = mask[idx]
            yield u_batch.transpose(1, 0, 2), x_batch.transpose(1, 0, 2), mask_batch.T

    def get_train_batch(self):
        return iter(self._iter_data(self.u_train, self.x_train, shuffle=True))

    def get_valid_batch(self):
        return iter(self._iter_data(self.u_valid, self.x_valid))

    def get_test_batch(self):
        return iter(self._iter_data(self.u_test, self.x_test,
                                    mask=self.mask_test))


class BlizzardIterator(object):
    def __init__(self, data, batch_size=None, nbatch=None,
		 start=0, end=None, shuffle=False, infinite_data=0,
                 pseudo_n=1000000):
        if (batch_size or nbatch) is None:
            raise ValueError("Either batch_size or nbatch should be given.")
        if (batch_size and nbatch) is not None:
            raise ValueError("Provide either batch_size or nbatch.")
        self.infinite_data = infinite_data
        if not infinite_data:
            self.start = start
            self.end = data.num_examples() if end is None else end
            if self.start >= self.end or self.start < 0:
                raise ValueError("Got wrong value for start %d." % self.start)
            self.nexp = self.end - self.start
            if nbatch is not None:
                self.batch_size = int(np.float(self.nexp / float(nbatch)))
                self.nbatch = nbatch
            elif batch_size is not None:
                self.batch_size = batch_size
                self.nbatch = int(np.float(self.nexp / float(batch_size)))
            self.shuffle = shuffle
        else:
            self.pseudo_n = pseudo_n
        self.data = data
        self.name = self.data.name

    def __iter__(self):
        if self.infinite_data:
            for i in range(self.pseudo_n):
                yield self.data.slices()
        else:
            start = self.start
            end = self.end - self.end % self.batch_size
            for idx in range(start, end, self.batch_size):
                x_batch = self.data.slices(idx, idx + self.batch_size)[0]
                y_batch = self.data.slices(idx + 1, idx + self.batch_size + 1)[0]
                mask_batch = np.ones((x_batch.shape[0], x_batch.shape[1]), dtype=x_batch.dtype)
                yield x_batch, y_batch, mask_batch

class IAMOnDBIterator(object):
    def __init__(self, data, batch_size=None, nbatch=None,
		 start=0, end=None, shuffle=False):
        if (batch_size or nbatch) is None:
            raise ValueError("Either batch_size or nbatch should be given.")
        if (batch_size and nbatch) is not None:
            raise ValueError("Provide either batch_size or nbatch.")
        self.start = start
        self.end = data.num_examples() if end is None else end
        if self.start >= self.end or self.start < 0:
            raise ValueError("Got wrong value for start %d." % self.start)
        self.nexp = self.end - self.start
        if nbatch is not None:
            self.batch_size = int(np.float(self.nexp / float(nbatch)))
            self.nbatch = nbatch
        elif batch_size is not None:
            self.batch_size = batch_size
            self.nbatch = int(np.float(self.nexp / float(batch_size)))
        self.shuffle = shuffle
        self.data = data
        self.name = self.data.name

    def __iter__(self):
        start = self.start
        end = self.end - self.end % self.batch_size
        for idx in range(start, end, self.batch_size):
            data_batch, mask_batch = self.data.slices(idx, idx + self.batch_size)
            x_batch = data_batch[:-1]; y_batch = data_batch[1:];
            #print(x_batch.shape);
            mask_batch = mask_batch[1:];
            yield x_batch, y_batch, mask_batch
from motion_utils import read_bvh

class DanceIterator(object):
    def __init__(self, dance_folder, fnames, seq_len, is_test = False, X_mean = None, X_std = None, ab = False):
        self.fnames = fnames
        self.ab = ab;
        self.dances = self.load_dances(dance_folder);
        self.seq_len = seq_len;
        self.in_frame = 171;
        self.X_mean = X_mean;
        self.X_std = X_std;
        self.is_test = is_test
        self.dances, self.mask = self.batchify_(self.dances, is_test, mean = self.X_mean, std = self.X_std);
        if (self.X_mean is None):
            self.dances = torch.stack(self.dances);
            self.mask = torch.stack(self.mask);
            self.X_mean = self.dances.mean();
            self.X_std = self.dances.std();
            self.dances = self.dances - self.X_mean;
            self.dances = self.dances / self.X_std;
    
    def load_dances(self, dance_folder):
        dances=[]
        for dance_file in self.fnames:
            dance=np.load(dance_folder+dance_file)
            dances.append(dance)
        return dances    
    
    def batchify_(self, dances, is_test=False, mean = None, std = None):
        seq_len = self.seq_len;
        dance_batch=[];
        mask_batch = []
        if (self.ab):
            Hip_index = read_bvh.joint_index['hip']
        for dance_idx in xrange(0, len(dances)):
            seq = torch.FloatTensor(dances[dance_idx]);
            if (self.ab == False):
                seq_diff = seq[1:] - seq[:-1];
            else:
                diff = seq[1:] - seq[:-1];
                seq_diff = seq[:-1];
                seq_diff[:, Hip_index * 3] = diff[:, Hip_index * 3];
                seq_diff[:, Hip_index * 3 + 2] = diff[:, Hip_index * 3 + 2];
            if (mean is not None):
                seq_diff = seq_diff - mean;
                seq_diff = seq_diff / std;
            #enable this when training walking and india else disable
            if (self.is_test == False):
                delta = 32;
            else:
                delta = seq_len;
            #delta = seq_len;
            for idx in xrange(0, seq_diff.size(0), delta):
                if (idx + 1 == seq_diff.size(0)):
                    continue;
                if (idx+seq_len > seq_diff.size(0)):
                    if (is_test == False):
                        break;
                    tmp = torch.zeros(seq_len - (seq_diff.size(0) - idx) , self.in_frame)
                    tmp = torch.cat([seq_diff[idx:], tmp], 0);
                    dance_batch.append(tmp);
                    mask_tmp = torch.zeros(seq_len - 1);
                    mask_tmp[:seq_diff.size(0) - idx - 1] = 1;
                    mask_batch.append(mask_tmp);
                else:
                    dance_batch.append(seq_diff[idx:idx+seq_len]);
                    mask_batch.append(torch.ones(seq_len - 1));
            
        return dance_batch, mask_batch;

    def get_batch(self, batch_size=1, test=False):
        dance = self.dances
        mask = self.mask;
        indices = np.arange(len(dance))
        if (test == False):
            np.random.shuffle(indices);
        for idx in xrange(0, len(dance), batch_size):
            end = min(idx + batch_size, len(dance));
            dance_batch = [];
            if (test == True):
                dance_batch = dance[idx];
                dance_batch = dance_batch.unsqueeze(0);
                mask_batch = mask[idx];
                mask_batch = mask_batch.unsqueeze(0);
            else:
                dance_batch = dance[indices[idx:end]];
                '''
                #augment the direction and position of the dance
                dance_batch = dance_batch.numpy()
                train_batch = []
                for i in range(batch_size):
                    T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
                    R=[0,1,0,(random.random()-0.5)*np.pi*2]
                    sample_seq = dance_batch[i];
                    sample_seq_augmented=read_bvh.augment_train_data(sample_seq, T, R)
                    train_batch=train_batch+[torch.from_numpy(sample_seq_augmented)]
                dance_batch = torch.stack(train_batch);
                '''
                mask_batch = mask[indices[idx:end]]
            #if (dance_batch.size(1) == 1):
            #    continue;
            x = dance_batch[:, :-1, :];
            y = dance_batch[:, 1:, :];
            x = x.permute(1,0,2);
            y = y.permute(1,0,2);

            x_mask = mask_batch
            x_mask = x_mask.permute(1,0);
            
            yield x, y, x_mask
