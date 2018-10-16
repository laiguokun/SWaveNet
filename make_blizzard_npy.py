from scipy.io import wavfile
import cPickle
import fnmatch
import os
data_dir = '/usr0/home/bohanl1/datasets/blizzard2013/'
list_len = 200
i = 0
l = []
files = []
for root, dir_names, file_names in os.walk(data_dir):
    for filename in fnmatch.filter(file_names, '*.wav'):
        files.append(os.path.join(root, filename))
#files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
n_frame = 0
for n, f in enumerate(files):
    sr, d = wavfile.read(f)
    l.append(d)
    #print(type(d))
    #print sr, len(d)
    #raw_input()
    n_frame += len(d)
    print('n_frames', n_frame, 'n_samples', n_frame/32000.)
    if len(l) >= list_len:
        print("Dumping at file %i of %i" % (n, len(files)))
        cPickle.dump(l, open("data_%i.npy" % i, mode="wb"))
        i += 1
        l = []
#dump last chunk
cPickle.dump(l, open("data_%i.npy" % i, mode="wb"))
