import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import theano.tensor as T

from iamondb_utils import fetch_iamondb

def tolist(arg):
    if type(arg) is not list:
        if isinstance(arg, tuple):
            return list(arg)
        else:
            return [arg]
    return arg


def totuple(arg):
    if type(arg) is not tuple:
        if isinstance(arg, list):
            return tuple(arg)
        else:
            return (arg,)
    return arg


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    This code has been implemented by Anne Archibald and has been discussed
    on the ML.
    Parameters
    ----------
    a : array-like
        The array to segment
    length : int
        The length of each frame
    overlap : int, optional
        The number of array elements by which the frames should overlap
    axis : int, optional
        The axis to operate on; if None, act on the flattened array
    end : {'cut', 'wrap', 'end'}, optional
        What to do with the last frame, if the array is not evenly
        divisible into pieces.
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value
    endvalue : object
        The value to use for end='pad'
    Examples
    --------
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    Notes
    -----
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    use as_strided
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap<0 or length<=0:
        raise ValueError("overlap must be nonnegative and length must be positive")

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'")
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)

def complex_to_real(X):
    """
    WRITEME
    Parameters
    ----------
    X : list of complex vectors
    Notes
    -----
    This function assumes X as 2D
    """
    new_X = []
    for i in range(len(X)):
        x = X[i]
        new_x = np.concatenate([np.real(x), np.imag(x)])
        new_X.append(new_x)
    return np.array(new_X)



class SequentialPrepMixin(object):
    """
    Preprocessing mixin for sequential data
    """
    def norm_normalize(self, X, avr_norm=None):
        """
        Unify the norm of each sequence in X
        Parameters
        ----------
        X       : list of lists or ndArrays
        avr_nom : Scalar
        """
        if avr_norm is None:
            avr_norm = 0
            for i in range(len(X)):
                euclidean_norm = np.sqrt(np.square(X[i].sum()))
                X[i] /= euclidean_norm
                avr_norm += euclidean_norm
            avr_norm /= len(X)
        else:
            X = [x[i] / avr_norm for x in X]
        return X, avr_norm

    def global_normalize(self, X, X_mean=None, X_std=None):
        """
        Globally normalize X into zero mean and unit variance
        Parameters
        ----------
        X      : list of lists or ndArrays
        X_mean : Scalar
        X_std  : Scalar
        Notes
        -----
        Compute varaince using the relation
        >>> Var(X) = E[X^2] - E[X]^2
        """
        if X_mean is None or X_std is None:
            X_len = np.array([len(x) for x in X]).sum()
            X_mean = np.array([x.sum() for x in X]).sum() / X_len
            X_sqr = np.array([(x**2).sum() for x in X]).sum() / X_len
            X_std = np.sqrt(X_sqr - X_mean**2)
            X = (X - X_mean) / X_std
        else:
            X = (X - X_mean) / X_std
        return (X, X_mean, X_std)

    def standardize(self, X, X_max=None, X_min=None):
        """
        Standardize X such that X \in [0, 1]
        Parameters
        ----------
        X     : list of lists or ndArrays
        X_max : Scalar
        X_min : Scalar
        """
        if X_max is None or X_min is None:
            X_max = np.array([x.max() for x in X]).max()
            X_min = np.array([x.min() for x in X]).min()
            X = (X - X_min) / (X_max - X_min)
        else:
            X = (X - X_min) / (X_max - X_min)
        return (X, X_max, X_min)

    def numpy_rfft(self, X):
        """
        Apply real FFT to X (numpy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.rfft(x) for x in X])
        return X

    def numpy_irfft(self, X):
        """
        Apply real inverse FFT to X (numpy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([np.fft.irfft(x) for x in X])
        return X

    def rfft(self, X):
        """
        Apply real FFT to X (scipy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.rfft(x) for x in X])
        return X

    def irfft(self, X):
        """
        Apply real inverse FFT to X (scipy)
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fftpack.irfft(x) for x in X])
        return X

    def stft(self, X):
        """
        Apply short-time Fourier transform to X
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.fft(x) for x in X])
        return X

    def istft(self, X):
        """
        Apply short-time Fourier transform to X
        Parameters
        ----------
        X     : list of lists or ndArrays
        """
        X = np.array([scipy.real(scipy.ifft(x)) for x in X])
        return X

    def fill_zero1D(self, x, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix
        Parameters
        ----------
        X       : list or 1D ndArray
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if mode == 'lefthand':
            new_x = np.concatenate([np.zeros((pad_len)), x])
        elif mode == 'righthand':
            new_x = np.concatenate([x, np.zeros((pad_len))])
        elif mode == 'random':
            new_x = np.concatenate(
                [np.zeros((pad_len)), x, np.zeros((pad_len))]
            )
        return new_x

    def fill_zero(self, X, pad_len=0, mode='righthand'):
        """
        Given variable lengths sequences,
        pad zeros w.r.t to the maximum
        length sequences and create a
        dense design matrix
        Parameters
        ----------
        X       : list of ndArrays or lists
        pad_len : integer
            if 0, we consider that output should be
            a design matrix.
        mode    : string
            Strategy to fill-in the zeros
            'righthand': pad the zeros at the right space
            'lefthand' : pad the zeros at the left space
            'random'   : pad the zeros with randomly
                         chosen left space and right space
        """
        if pad_len == 0:
            X_max = np.array([len(x) for x in X]).max()
            new_X = np.zeros((len(X), X_max))
            for i, x in enumerate(X):
                free_ = X_max - len(x)
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((free_)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((free_))], axis=1)
                elif mode == 'random':
                    j = np.random.randint(free_)
                    new_x = np.concatenate(
                        [np.zeros((j)), x, np.zeros((free_ - j))],
                        axis=1
                    )
                new_X[i] = new_x
        else:
            new_X = []
            for x in X:
                if mode == 'lefthand':
                    new_x = np.concatenate([np.zeros((pad_len)), x], axis=1)
                elif mode == 'righthand':
                    new_x = np.concatenate([x, np.zeros((pad_len))], axis=1)
                elif mode == 'random':
                    new_x = np.concatenate(
                        [np.zeros((pad_len)), x, np.zeros((pad_len))],
                         axis=1
                    )
                new_X.append(new_x)
        return new_X

    def reverse(self, X):
        """
        Reverse each sequence of X
        Parameters
        ----------
        X       : list of ndArrays or lists
        """
        new_X = []
        for x in X:
            new_X.append(x[::-1])
        return new_X


class Data(object):
    """
    Abstract class for data
    Parameters
    ----------
    .. todo::
    """
    def __init__(self, name=None, path=None, multi_process=0):
        self.name = name
        self.data = self.load(path)
        self.multi_process = multi_process
        if multi_process > 0:
            self.queue = Queue(2**15)
            processes = [None] * multi_process
            for mid in range(multi_process):
                processes[mid] = Process(target=self.multi_process_slices,
                                         args=(mid,))
                processes[mid].start()

    def multi_process_slices(self, mid=-1):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.multi_process_slices.")

    def load(self, path):
        return np.load(path)

    def slices(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.slices.")

    def num_examples(self):
        return max(mat.shape[0] for mat in self.data)

    def theano_vars(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement Data.theano_vars.")


class TemporalSeries(Data):
    """
    Abstract class for temporal data.
    We use TemporalSeries when the data contains variable length
    seuences, otherwise, we use DesignMatrix.
    Parameters
    ----------
    .. todo::
    """
    def slices(self, start, end):
        return (mat[start:end].swapaxes(0, 1)
                for mat in self.data)

    def create_mask(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        for i, sample_len in enumerate(samples_len):
            mask[:sample_len, i] = 1.
        return mask

    def zero_pad(self, batch):
        max_sample_len = max(len(sample) for sample in batch)
        rval = np.zeros((len(batch), max_sample_len, batch[0].shape[-1]),
                        dtype=batch[0].dtype)
        for i, sample in enumerate(batch):
            rval[i, :len(sample)] = sample
        return rval.swapaxes(0, 1)

    def create_mask_and_zero_pad(self, batch):
        samples_len = [len(sample) for sample in batch]
        max_sample_len = max(samples_len)
        mask = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        if batch[0].ndim == 1:
            rval = np.zeros((max_sample_len, len(batch)), dtype=batch[0].dtype)
        else:
            rval = np.zeros((max_sample_len, len(batch), batch[0].shape[1]),
                            dtype=batch[0].dtype)
        for i, (sample, sample_len) in enumerate(zip(batch, samples_len)):
            mask[:sample_len, i] = 1.
            if batch[0].ndim == 1:
                rval[:sample_len, i] = sample
            else:
                rval[:sample_len, i, :] = sample
        return rval, mask



class IAMOnDB(TemporalSeries, SequentialPrepMixin):
    """
    IAMOnDB dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, prep='none', cond=False, X_mean=None, X_std=None,
                 bias=None, **kwargs):

        self.prep = prep
        self.cond = cond
        self.X_mean = X_mean
        self.X_std = X_std
        self.bias = bias

        super(IAMOnDB, self).__init__(**kwargs)

    def load(self, data_path):

        if self.name == "train":
            X, y, _, _ = fetch_iamondb(data_path)
            print("train")
            print(len(X))
            print(len(y))
        elif self.name == "valid":
            _, _, X, y = fetch_iamondb(data_path)
            print("valid")
            print(len(X))
            print(len(y))

        raw_X = X
        raw_X0 = []
        offset = True
        raw_new_X = []

        for item in raw_X:
            if offset:
                raw_X0.append(item[1:, 0])
                raw_new_X.append(item[1:, 1:] - item[:-1, 1:])
            else:
                raw_X0.append(item[:, 0])
                raw_new_X.append(item[:, 1:])

        raw_new_X, self.X_mean, self.X_std = self.global_normalize(raw_new_X, self.X_mean, self.X_std)
        new_x = []

        for n in range(raw_new_X.shape[0]):
            new_x.append(np.concatenate((raw_X0[n][:, None], raw_new_X[n]),
                                        axis=-1).astype('float32'))
        new_x = np.array(new_x)

        if self.prep == 'none':
            X = np.array(raw_X)

        if self.prep == 'normalize':
            X = new_x
            print X[0].shape
        elif self.prep == 'standardize':
            X, self.X_max, self.X_min = self.standardize(raw_X)

        self.labels = [np.array(y)]

        return [X]

    def theano_vars(self):

        if self.cond:
            return [T.ftensor3('x'), T.fmatrix('mask'),
                    T.ftensor3('y'), T.fmatrix('label_mask')]
        else:
            return [T.ftensor3('x'), T.fmatrix('mask')]

    def theano_test_vars(self):
        return [T.ftensor3('y'), T.fmatrix('label_mask')]

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        label_batches = [mat[start:end] for mat in self.labels]
        len_batches = len(batches[0].shape)
        if(len_batches <= 1):
            mask = self.create_mask(batches[0])
        else:
            mask = self.create_mask(batches[0].swapaxes(0, 1))
        batches = [self.zero_pad(batch) for batch in batches]
        len_label_batches = len(label_batches[0].shape)
        if(len_label_batches <= 1):
            label_mask = self.create_mask(label_batches[0])
        else:
            label_mask = self.create_mask(label_batches[0].swapaxes(0, 1))
        label_batches = [self.zero_pad(batch) for batch in label_batches]

        if self.cond:
            return totuple([batches[0], mask, label_batches[0], label_mask])
        else:
            return totuple([batches[0], mask])

    def generate_index(self, X):

        maxlen = np.array([len(x) for x in X]).max()
        idx = np.arange(maxlen)

        return idx

if __name__ == "__main__":

    data_path = 'datasets/iamondb/'
    iamondb = IAMOnDB(name='train',
                      prep='normalize',
                      cond=False,
                      path=data_path)

    batch = iamondb.slices(start=0, end=10826)
    X = iamondb.data[0]
    sub_X = X

    for item in X:
        max_x = np.max(item[:,1])
        max_y = np.max(item[:,2])
        min_x = np.min(item[:,1])
        min_y = np.min(item[:,2])

    print np.max(max_x)
    print np.max(max_y)
    print np.min(min_x)
    print np.min(min_y)
