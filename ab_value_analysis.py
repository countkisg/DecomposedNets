from skimage import color
import numpy as np
import scipy.misc
from datasets import CelebA, BigDataset
from progressbar import ETA, Bar, Percentage, ProgressBar
import matplotlib.pyplot as plt
def find_closest_idx(ll, value):
    return np.abs(ll-value).argmin()

color.colorconv.lab_ref_white = np.array([0.96422, 1.0, 0.82521])

q_shape = (20, 30, 30)
l_lin = np.linspace(0, 99, q_shape[0])
a_lin = np.linspace(-128, 128, q_shape[1])
b_lin = np.linspace(-128, 128, q_shape[2])

batch_size = 1

celeba = CelebA().train
q = np.zeros(shape=q_shape, dtype=np.float32)
iterations = celeba.num_examples / batch_size

widgets = ["Quantize #|", Percentage(), Bar(), ETA()]
pbar = ProgressBar(maxval=celeba.num_examples, widgets=widgets)
pbar.start()
for i_i in range(iterations):
    img, flag = celeba.next_sequencial_batch(batch_size)
    lab = color.rgb2lab(np.reshape(img[0], newshape=celeba.image_shape))
    for j in range(celeba.image_shape[1]):
        for i in range(celeba.image_shape[0]):
            id_l = find_closest_idx(l_lin, lab[i, j, 0])
            id_a = find_closest_idx(a_lin, lab[i, j, 1])
            id_b = find_closest_idx(b_lin, lab[i, j, 2])
            q[id_l, id_a, id_b] += 1
    pbar.update(i_i)
    if i_i == 100: break
    if False == flag: break

np.save('quan', q)

