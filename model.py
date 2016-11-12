import tensorflow as tf
from distribution import Product, Distribution, Gaussian, Categorical, Bernoulli
from ops import BatchNorm, conv2d, deconv2d, linear, binary_crossentropy, lrelu
from math import ceil
import scipy.io
from vgg import vgg_net
class InfoGAN(object):
    def __init__(self, output_dist, latent_spec, batch_size, image_shape, network_type, vgg_path):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product(
            [x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        # mnist dataset
        self.d_e_bn0 = BatchNorm(self.batch_size, name='d_e_bn0')
        self.d_e_bn1 = BatchNorm(self.batch_size, name='d_e_bn1')
        self.d_e_bn2 = BatchNorm(self.batch_size, name='d_e_bn2')

        self.ge_bn0 = BatchNorm(self.batch_size, name='ge_bn0')
        self.ge_bn1 = BatchNorm(self.batch_size, name='ge_bn1')
        self.ge_bn2 = BatchNorm(self.batch_size, name='ge_bn2')
        # CelebA dataset
        self.d_face_bn0 = BatchNorm(self.batch_size, name='d_face_bn0')
        self.d_face_bn1 = BatchNorm(self.batch_size, name='d_face_bn1')
        self.d_face_bn2 = BatchNorm(self.batch_size, name='d_face_bn2')

        self.ge_face_bn0 = BatchNorm(self.batch_size, name='ge_face_bn0')
        self.ge_face_bn1 = BatchNorm(self.batch_size, name='ge_face_bn1')
        self.ge_face_bn2 = BatchNorm(self.batch_size, name='ge_face_bn2')
        self.ge_face_bn3 = BatchNorm(self.batch_size, name='ge_face_bn3')
        self.ge_face_bn4 = BatchNorm(self.batch_size, name='ge_face_bn4')

        # load pre-trained vgg-19 model
        self.vgg_model = scipy.io.loadmat(vgg_path)

    def discriminate(self, x_var, reuse=None):
        if self.network_type == "mnist":
            with tf.variable_scope("d_net", reuse=reuse):
                input = tf.reshape(x_var, shape=[-1]+list(self.image_shape))
                h0 = lrelu(self.d_e_bn0(conv2d(input, 64, k_h=4, k_w=4, name='d_e_conv0')))
                h1 = lrelu(self.d_e_bn1(conv2d(h0, 128, k_w=4, k_h=4, name='d_e_conv1')))
                h2 = lrelu(linear(tf.reshape(h1, [self.batch_size, -1]), 1024, name='d_e_linear0'))

                discriminator_template = linear(h2, 1, name='d_e_real_prob')
                encoder_template = linear(lrelu(linear(h2, 128)), self.reg_latent_dist.dist_flat_dim, name='d_e_noise_code')

                d_out = tf.identity(discriminator_template, name='d_out')
                d = tf.nn.sigmoid(d_out[:, 0])
                reg_dist_flat = tf.identity(encoder_template, name='reg_dist_flat')
                reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
                return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

        elif self.network_type == 'celeba':
            with tf.variable_scope('d_net', reuse=reuse):
                img = tf.reshape(x_var, shape=[-1]+list(self.image_shape))
                h0 = lrelu(conv2d(img, output_dim=64, k_w=4, k_h=4, name='d_face_conv0'))
                h1 = lrelu(self.d_face_bn0(conv2d(h0, output_dim=128, k_w=3, k_h=3, name='d_face_conv1')))
                h2 = lrelu(self.d_face_bn1(conv2d(h1, output_dim=256, k_w=3, k_h=3, name='d_face_conv2')))
                h3 = lrelu(self.d_face_bn2(conv2d(h2, output_dim=512, k_h=2, k_w=2, name='d_face_conv3')))
                h4 = tf.reshape(h3, [self.batch_size, -1])
                discriminator_template = linear(h4, 1, name='d_face_real_prob')
                encoder_template = linear(lrelu(linear(h4, 126, name='d_face_linear0')), self.reg_latent_dist.dist_flat_dim,
                                          name='d_face_noise_code')

                d_out = tf.identity(discriminator_template, name='d_out')
                d = tf.nn.sigmoid(d_out[:, 0])
                reg_dist_flat = tf.identity(encoder_template, name='reg_dist_flat')
                reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
                return d, reg_dist_info
        else:
            raise NotImplementedError

    def generate(self, z_var, reuse=None):
        if self.network_type == 'mnist':
            with tf.variable_scope('g_net', reuse=reuse):
                noise_code = tf.reshape(z_var, [self.batch_size, self.latent_dist.dim])
                #h0 = lrelu(self.ge_bn0(tf.reshape(linear(noise_code, 1024), ), name='ge_linear0')
                h0 = lrelu(linear(noise_code, 1024, name='ge_linear0'))
                h1 = lrelu(self.ge_bn1(tf.reshape(linear(h0, 7*7*128, name='ge_linear1'), [self.batch_size,7,7,128])))
                h2 = lrelu(self.ge_bn2(deconv2d(h1, output_shape=[self.batch_size,14, 14, 64], k_h=4,k_w=4, name='ge_deconv0')))
                x_dist_flat = tf.reshape(deconv2d(h2, output_shape=[self.batch_size, 28, 28, 1], name='ge_deconv1'),
                                         shape=[self.batch_size, -1])
                x_dist_info = self.output_dist.activate_dist(x_dist_flat)
                return self.output_dist.sample(x_dist_info), x_dist_info
        elif self.network_type == 'celeba':
            with tf.variable_scope('g_net', reuse=reuse):
                noise_code = tf.reshape(z_var, [self.batch_size, self.latent_dist.dim])
                h0 = lrelu(linear(noise_code, int(ceil(self.image_shape[0]/32.)) * int(ceil(self.image_shape[1]/32.)) * 256,
                                name='ge_face_linear0'))
                h1 = lrelu(self.ge_face_bn0(deconv2d(tf.reshape(h0, shape=[self.batch_size,
                                                                           int(ceil(self.image_shape[0] / 32.)),
                                                                           int(ceil(self.image_shape[1] / 32.)),
                                                                           256]),
                                                     output_shape=[self.batch_size,
                                                     int(ceil(self.image_shape[0]/16.)),
                                                     int(ceil(self.image_shape[1]/16.)), 128], k_h=4, k_w=4,
                                                     name='ge_face_deconv0')))
                h2 = lrelu(self.ge_face_bn1(deconv2d(h1, output_shape=[self.batch_size, int(ceil(self.image_shape[0]/8.)),
                                    int(ceil(self.image_shape[1]/8.)), 64], k_h=3, k_w=3,
                                    name='ge_face_deconv1')))
                h3 = lrelu(self.ge_face_bn2(deconv2d(h2, output_shape=[self.batch_size, int(ceil(self.image_shape[0]/4.)),
                                    int(ceil(self.image_shape[1]/4.)), 32], k_h=3, k_w=3,
                                    name='ge_face_deconv2')))
                h4 = lrelu(self.ge_face_bn3(deconv2d(h3, output_shape=[self.batch_size, int(ceil(self.image_shape[0]/2.)),
                                                      int(ceil(self.image_shape[1]/2.)), 16], k_h=2, k_w=2,
                                    name='ge_face_deconv3')))
                x_dist_flat = tf.nn.tanh(self.ge_face_bn4(deconv2d(h4, output_shape=[self.batch_size] + self.image_shape, k_w=2, k_h=2,
                                    name='ge_face_deconv4')))
                x_dist_info = self.output_dist.activate_dist(x_dist_flat)
                return x_dist_flat, x_dist_info
        else:
            raise NotImplementedError
    def vgg_generator(self, conv_z, reuse=None):
        with tf.variable_scope('c_net', reuse=reuse):
            return


    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)




