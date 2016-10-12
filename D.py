from datasets import CelebA, BigDataset
import matplotlib.pyplot as plt
import numpy as np
data = CelebA()
x, _ = data.train.next_batch(100)
plt.imshow(np.reshape(x[0], newshape=data.image_shape))
tmp =1