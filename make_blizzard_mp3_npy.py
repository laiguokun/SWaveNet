from scipy.io import wavfile
import cPickle
import fnmatch
import os


from subprocess import Popen, PIPE
import numpy as np

def decode (fname):
    # If you are on Windows use full path to ffmpeg.exe
    cmd = ["ffmpeg", "-i", fname, "-f", "wav", "-ar", "16000", '-']
    # If you are on W add argument creationflags=0x8000000 to prevent another console window jumping out
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    data = p.communicate()[0]
    return np.fromstring(data[data.find("data")+4:], np.int16)


data_dir = '/usr0/home/bohanl1/datasets/blizzard2013/Lessac_Blizzard2013_CatherineByers_train/unsegmented'
list_len = 200
i = 0
l = []
files = []
for root, dir_names, file_names in os.walk(data_dir):
    for filename in fnmatch.filter(file_names, '*.mp3'):
        files.append(os.path.join(root, filename))
#files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
n_frame = 0
for n, f in enumerate(files):
    #sr, d = wavfile.read(f)
    d = decode(f)
    l.append(d)
    """
    print(d.shape)
    print(min(d), max(d))
    print(type(d))
    print len(d)
    raw_input()
    """
    n_frame += len(d)
    print('n_frames', n_frame, 'n_samples', n_frame/32000., 'length', '%f hours' % (1.*n_frame/64000/3600))
    if len(l) >= list_len:
        print("Dumping at file %i of %i" % (n, len(files)))
        cPickle.dump(l, open("data_%i.npy" % i, mode="wb"))
        i += 1
        l = []
#dump last chunk
cPickle.dump(l, open("data_%i.npy" % i, mode="wb"))
