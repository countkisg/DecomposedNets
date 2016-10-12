import numpy as np
import matplotlib.pyplot as plt
non_reg = [10]*62
reg = [0,0,0,1,0,0,0,0,0,0, 0,0]
z = tf.constant([non_reg+reg]*128,dtype=tf.float32)
fake_x, _ = self.model.generate(z, reuse=True)
fake = sess.run(fake_x)
plt.imshow(np.reshape(fake[0], (28,28)))