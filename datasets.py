from tensorflow.examples.tutorials import mnist
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.misc

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
    def __init__(self, filespath, labels=None, image_shape=None, image_dim=None):
        #self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._filespath = filespath
        self._num_examples = len(filespath)
        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._image_shape = image_shape
        self._image_dim = image_dim

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __normalize(self, arr):
        if arr.max() > 1.0:
            arr /= 255.0
        return arr
    def __read_images(self, start, end, batch_size):
        result = np.zeros(shape=[batch_size]+list(self._image_shape))
        for i in range(batch_size):
            path = self._filespath[start+i]
            im = scipy.misc.imread(path)
            result[i] = self.__normalize(scipy.misc.imresize(im, size=self._image_shape).astype(np.float32))
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
        end = self._index_in_epoch
        if self._labels is None:
            return self.__read_images(start, end, batch_size), None
        else:
            return NotImplementedError

class CelebA(object):
    def __init__(self):
        self.image_dim = 110*88*3
        self.image_shape = [110, 88, 3]

        self._data_directory = 'img_align_celeba/'
        self._onlyfiles = [self._data_directory+f for f in listdir(self._data_directory) if isfile(join(self._data_directory, f))]
        self.train = BigDataset(filespath=self._onlyfiles, image_shape=self.image_shape, image_dim=self.image_dim)


