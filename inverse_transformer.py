from scipy import ndimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
TINY = tf.constant(1e-10, dtype=tf.float32)
def inverse_theta(thetas):
    assert isinstance(thetas, list)
    assert len(thetas) == 6
    one = tf.constant(1, dtype=tf.float32)
    itheta0 = tf.div(one, TINY + thetas[0])
    itheta1 = thetas[1]
    itheta2 = tf.div(-thetas[2], thetas[0]+TINY)
    itheta3 = thetas[3]
    itheta4 = tf.div(one, TINY + thetas[4])
    itheta5 = tf.div(-thetas[5], TINY+thetas[4])

    return tf.reshape(tf.pack([itheta0, itheta1, itheta2, itheta3, itheta4, itheta5]), shape=(6,))

def inverse_transformer(Us, theta, out_size, num_batch, name='InverseTransformer' ):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.pack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.pack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.pack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(0, [x_t_flat, y_t_flat, ones])
            return grid

    def _inverse_transform(input_dim, theta, out_size, sub_nets):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.pack([num_batch]))
            grid = tf.reshape(grid, tf.pack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.batch_matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.pack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        # assert isinstance(Us, list)
        # assert isinstance(thetas, list)
        # assert len(Us) == len(thetas)

        #sub_nets = len(Us)
        # assert sub_nets > 0
        thetas = tf.zeros([num_batch, 6]) + theta
        return _inverse_transform(Us, thetas, out_size, 1)


if __name__ == '__main__':
    im = ndimage.imread('cat.jpg')
    im = im / 255.
    im = im.reshape(1, 1200, 1600, 3)
    im = im.astype('float32')

    # %% Let the output size of the transformer be half the image size.
    out_size = (200, 400)

    # %% Simulate batch
    batch = np.append(im, im, axis=0)
    batch = np.append(batch, im, axis=0)
    num_batch = 3

    x = tf.cast(batch, 'float32')

    # %% Create localisation network and convolutional layer
    with tf.variable_scope('spatial_transformer_0'):
        # %% Create a fully-connected layer with 6 output nodes
        n_fc = 6
        W_fc1 = tf.Variable(tf.zeros([1200 * 1600 * 3, n_fc]), name='W_fc1')

        # %% Zoom into the image
        initial = np.array([[0.5, 0, 0.5], [0, 0.5, 0.2]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        theta0 = tf.Variable(initial_value=np.array([0.5]), name='theta1', dtype=tf.float32)
        theta1 = tf.Variable(initial_value=np.array([0]), name='theta2', dtype=tf.float32)
        theta2 = tf.Variable(initial_value=np.array([0.5]), name='theta3', dtype=tf.float32)
        theta3 = tf.Variable(initial_value=np.array([0]), name='theta4', dtype=tf.float32)
        theta4 = tf.Variable(initial_value=np.array([0.5]), name='theta5', dtype=tf.float32)
        theta5 = tf.Variable(initial_value=np.array([0.2]), name='theta6', dtype=tf.float32)
        theta_list = [theta0, theta1, theta2, theta3, theta4, theta5]
        theta = tf.reshape(tf.pack(theta_list), shape=(6,))
        h_fc1 = tf.matmul(tf.zeros([num_batch, 1200 * 1600 * 3]), W_fc1) + theta
        h_trans = inverse_transformer(x, h_fc1, out_size)

    # %% Run session
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        y = sess.run(h_trans)
        plt.imshow(y[0])

        theta = inverse_theta(theta_list)
        h_fc2 = tf.zeros([num_batch, 6]) + theta
        y = tf.cast(y, dtype=tf.float32)
        inv_trans = inverse_transformer(y, h_fc2, [1200, 1600])
        y = sess.run(inv_trans)
        plt.imshow(y[0])
        plt.imshow(y[0])