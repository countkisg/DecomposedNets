import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.Saver()
    tf.restore(sess, 'ckt/mnist/mnist_2016_08_19_10_29_15/mnist_2016_08_19_10_29_15_100.ckpt')