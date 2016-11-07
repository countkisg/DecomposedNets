from tensorflow.examples.tutorials import mnist
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.misc
from skimage.color import rgb2gray
class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class MnistDataset(object):
    def __init__(self):
        data_directory = "mnist"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation

        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data
class BigDataset(Dataset):
    def __init__(self, filespath, labels=None, image_shape=None, image_dim=None, offset=[28,-20,28,-30]):
        #self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._filespath = filespath
        self._num_examples = len(filespath)
        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._image_shape = image_shape
        self._image_dim = image_dim
        self._sequencial_index_in_epoch = 0
        self._offset = offset

    @property
    def labels(self):
        return self._labels
    @property
    def image_shape(self):
        return self._image_shape
    @property
    def off_set(self):
        return self._offset

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __normalize(self, arr, keep_value=False):
        if False == keep_value:
            if arr.max() > 1.0:
                arr = arr/128.0 - 1.
            return arr
        else:
            return arr/255.0
    def __read_images(self, start, batch_size, keep_value=False):
        result = np.zeros(shape=[batch_size]+list(self._image_shape[0:2]))
        for i in range(batch_size):
            path = self._filespath[start+i]
            im = np.reshape(rgb2gray(scipy.misc.imread(path)), newshape=(218,178,1))
            croped_im = scipy.misc.imresize(im[self._offset[0]:self._offset[1], self._offset[2]:self._offset[3], 0],
                                            size=self._image_shape[0:2], interp='bicubic').astype(np.float32)
            #croped = im.astype(np.float32)[28:-20, 28:-30,:]
            result[i] = self.__normalize(croped_im, keep_value)
            #result[i] = self.__normalize(scipy.misc.imresize(im, size=(48,56,3)).astype(np.float32))
        return np.reshape(result, newshape=[batch_size, -1])


    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            #self._filespath = self._filespath[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        if self._labels is None:
            return self.__read_images(start, batch_size), None
        else:
            return NotImplementedError
    def next_sequencial_batch(self, batch_size):
        start = self._sequencial_index_in_epoch
        self._sequencial_index_in_epoch += batch_size
        if self._sequencial_index_in_epoch >= self._num_examples-1:
            self._sequencial_index_in_epoch=0
            return self.__read_images(0, batch_size, keep_value=True), False
        #return self.__read_images(start, batch_size), self._filespath[start:start+batch_size]
        return self.__read_images(start, batch_size, keep_value=True), True


class CelebA(object):
    def __init__(self, small_size=None):
        self.image_dim = 98*80*1
        self.image_shape = [98, 80, 1]
        self._offset = [28,-20,28,-30]
        self._orignal_shape = [218, 178, 3]
        self._data_directory = 'img_align_celeba/'
        self._onlyfiles = np.sort([self._data_directory+f for f in listdir(self._data_directory) if isfile(join(self._data_directory, f))])
        if small_size:
            self._onlyfiles = self._onlyfiles[0:small_size]

        self.train = BigDataset(filespath=self._onlyfiles, image_shape=self.image_shape, image_dim=self.image_dim,
                                offset=self._offset)

    def regenerate_landmarks(self, path='list_landmarks_align_celeba.txt'):
        lm = np.genfromtxt(path, delimiter=',')
        lm = lm[:, 1:-1] - self._offset[0]
        new_x_radio = float(self.image_shape[1]) / (self._orignal_shape[1] - self._offset[0] + self._offset[1])
        new_y_radio = float(self.image_shape[0]) / (self._orignal_shape[0] - self._offset[2] + self._offset[3])
        lm[:, 0::2] = lm[:, 0::2] * new_x_radio
        lm[:, 1::2] = lm[:, 1::2] * new_y_radio
        return lm.astype(int)



