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
#

data = CelebA()
lm = data.regenerate_landmarks()
print lm[0]
off = data.train.off_set
path = 'img_align_celeba/100000.jpg'
im = scipy.misc.imread(path)
im = scipy.misc.imresize(im[off[0]:off[1], off[2]:off[3], :],
                                size=[98, 80, 3], interp='bicubic').astype(np.float32)
plt.imshow(im)
plt.imshow(im)
#
# plt.imshow(np.reshape(ims, newshape=[128,128,3]))