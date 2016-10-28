from datasets import CelebA, BigDataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from vgg import vgg_net

# path = '../imagenet-vgg-verydeep-19.mat'
# im = scipy.misc.imread('100000.jpg').astype(np.float32)
# im = np.reshape(im, newshape=(-1,) + im.shape)
# data = scipy.io.loadmat(path)
# net, _ = vgg_net(data, im)
# with tf.Session() as sess:
#     fea = sess.run(net)
#     plt.imshow(fea['relu2_1'])
#     plt.imshow(fea['relu2_1'])
#     plt.imshow(fea['relu2_1'])

data = CelebA().train
ims = data.next_batch(1)
plt.imshow(np.reshape(ims, newshape=[128,128,3]))

plt.imshow(np.reshape(ims, newshape=[128,128,3]))