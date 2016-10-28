import tensorflow as tf
from distribution import Product, Distribution, Gaussian, Categorical, Bernoulli
from ops import BatchNorm, conv2d, deconv2d, linear, binary_crossentropy, lrelu, elm
import numpy as np
from inverse_transformer import inverse_theta, inverse_transformer
from transformer import  transformer
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

        # mnist network
        self.d_e_bn0 = BatchNorm(self.batch_size, name='d_e_bn0')
        self.d_e_bn1 = BatchNorm(self.batch_size, name='d_e_bn1')
        self.d_e_bn2 = BatchNorm(self.batch_size, name='d_e_bn2')

        self.ge_bn0 = BatchNorm(self.batch_size, name='ge_bn0')
        self.ge_bn1 = BatchNorm(self.batch_size, name='ge_bn1')
        self.ge_bn2 = BatchNorm(self.batch_size, name='ge_bn2')
        # decomposed mnist network
        self.d_sn_0_bn0 = BatchNorm(self.batch_size, name='d_sn_0_bn0')
        self.d_sn_0_bn1 = BatchNorm(self.batch_size, name='d_sn_0_bn1')
        self.d_sn_0_bn2 = BatchNorm(self.batch_size, name='d_sn_0_bn2')

        self.d_sn_1_bn0 = BatchNorm(self.batch_size, name='d_sn_1_bn0')
        self.d_sn_1_bn1 = BatchNorm(self.batch_size, name='d_sn_1_bn1')
        self.d_sn_1_bn2 = BatchNorm(self.batch_size, name='d_sn_1_bn2')

        self.d_sn_2_bn0 = BatchNorm(self.batch_size, name='d_sn_2_bn0')
        self.d_sn_2_bn1 = BatchNorm(self.batch_size, name='d_sn_2_bn1')
        self.d_sn_2_bn2 = BatchNorm(self.batch_size, name='d_sn_2_bn2')

        self.ge_sn_0_bn0 = BatchNorm(self.batch_size, name='ge_sn_0_bn0')
        self.ge_sn_0_bn1 = BatchNorm(self.batch_size, name='ge_sn_0_bn1')
        self.ge_sn_0_bn2 = BatchNorm(self.batch_size, name='ge_sn_0_bn2')

        self.ge_sn_1_bn0 = BatchNorm(self.batch_size, name='ge_sn_1_bn0')
        self.ge_sn_1_bn1 = BatchNorm(self.batch_size, name='ge_sn_1_bn1')
        self.ge_sn_1_bn2 = BatchNorm(self.batch_size, name='ge_sn_1_bn2')

        self.ge_sn_2_bn0 = BatchNorm(self.batch_size, name='ge_sn_2_bn0')
        self.ge_sn_2_bn1 = BatchNorm(self.batch_size, name='ge_sn_2_bn1')
        self.ge_sn_2_bn2 = BatchNorm(self.batch_size, name='ge_sn_2_bn2')
        # thetas
        # self.theta0_0 = tf.Variable(initial_value=np.array([0.5]), name='theta0_0', dtype=tf.float32)
        # self.theta0_1 = tf.Variable(initial_value=np.array([0]), name='theta0_1', dtype=tf.float32, trainable=True)
        # self.theta0_2 = tf.Variable(initial_value=np.array([0.5]), name='theta0_2', dtype=tf.float32)
        # self.theta0_3 = tf.Variable(initial_value=np.array([0]), name='theta0_3', dtype=tf.float32, trainable=True)
        # self.theta0_4 = tf.Variable(initial_value=np.array([0.5]), name='theta0_4', dtype=tf.float32)
        # self.theta0_5 = tf.Variable(initial_value=np.array([0.2]), name='theta0_5', dtype=tf.float32)
        # self.theta0_list = [self.theta0_0, self.theta0_1, self.theta0_2, self.theta0_3, self.theta0_4, self.theta0_5]
        # self.theta0 = tf.reshape(tf.pack(self.theta0_list), shape=(6,))
        # self.theta1_0 = tf.Variable(initial_value=np.array([0.5]), name='theta1_0', dtype=tf.float32)
        # self.theta1_1 = tf.Variable(initial_value=np.array([0]), name='theta1_1', dtype=tf.float32, trainable=True)
        # self.theta1_2 = tf.Variable(initial_value=np.array([0.5]), name='theta1_2', dtype=tf.float32)
        # self.theta1_3 = tf.Variable(initial_value=np.array([0]), name='theta1_3', dtype=tf.float32, trainable=True)
        # self.theta1_4 = tf.Variable(initial_value=np.array([0.5]), name='theta1_4', dtype=tf.float32)
        # self.theta1_5 = tf.Variable(initial_value=np.array([0.2]), name='theta1_5', dtype=tf.float32)
        # self.theta1_list = [self.theta1_0, self.theta1_1, self.theta1_2, self.theta1_3, self.theta1_4, self.theta1_5]
        # self.theta1 = tf.reshape(tf.pack(self.theta1_list), shape=(6,))
        # self.theta2_0 = tf.Variable(initial_value=np.array([0.5]), name='theta2_0', dtype=tf.float32)
        # self.theta2_1 = tf.Variable(initial_value=np.array([0]), name='theta2_1', dtype=tf.float32, trainable=True)
        # self.theta2_2 = tf.Variable(initial_value=np.array([0.5]), name='theta2_2', dtype=tf.float32)
        # self.theta2_3 = tf.Variable(initial_value=np.array([0]), name='theta2_3', dtype=tf.float32, trainable=True)
        # self.theta2_4 = tf.Variable(initial_value=np.array([0.5]), name='theta2_4', dtype=tf.float32)
        # self.theta2_5 = tf.Variable(initial_value=np.array([0.2]), name='theta2_5', dtype=tf.float32)
        # self.theta2_list = [self.theta2_0, self.theta2_1, self.theta2_2, self.theta2_3, self.theta2_4, self.theta2_5]
        # self.theta2 = tf.reshape(tf.pack(self.theta2_list), shape=(6,))
        # celeba
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
                return
                # input = tf.reshape(x_var, shape=[-1]+list(self.image_shape))
                # h0 = lrelu(self.d_e_bn0(conv2d(input, 64, k_h=4, k_w=4, name='d_e_conv0')))
                # h1 = lrelu(self.d_e_bn1(conv2d(h0, 128, k_w=4, k_h=4, name='d_e_conv1')))
                # h2 = lrelu(linear(tf.reshape(h1, [self.batch_size, -1]), 1024, name='d_e_linear0'))
                #
                # discriminator_template = linear(h2, 1, name='d_e_real_prob')
                # encoder_template = linear(lrelu(linear(h2, 128)), self.reg_latent_dist.dist_flat_dim, name='d_e_noise_code')
                #
                # d_out = tf.identity(discriminator_template, name='d_out')
                # d = tf.nn.sigmoid(d_out[:, 0])
                # reg_dist_flat = tf.identity(encoder_template, name='reg_dist_flat')
                # reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
                # return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat
        elif self.network_type == 'de_mnist':
            with tf.variable_scope('d_net', reuse=reuse):
                return
                # img = tf.reshape(x_var, shape=[-1] + list(self.image_shape))
                # # de compose image
                # out_size = [28, 28]
                # img0 = tf.reshape(transformer(img, self.theta0, out_size, self.batch_size), [self.batch_size]+out_size+[1])
                # img1 = tf.reshape(transformer(img, self.theta1, out_size, self.batch_size), [self.batch_size]+out_size+[1])
                # img2 = tf.reshape(transformer(img, self.theta2, out_size, self.batch_size), [self.batch_size]+out_size+[1])
                # # sub net 0
                # sn_0_0 = lrelu(self.d_sn_0_bn0(conv2d(img0, 32, k_h=4, k_w=4, name='d_sn_0_conv0')))
                # sn_0_1 = lrelu(self.d_sn_0_bn1(conv2d(sn_0_0, 64, k_w=4, k_h=4, name='d_sn_0_conv1')))
                # sn_0_2 = lrelu(linear(tf.reshape(sn_0_1, [self.batch_size, -1]), 256, name='d_sn0_linear0'))
                # # sub net 1
                # sn_1_0 = lrelu(self.d_sn_1_bn0(conv2d(img1, 32, k_h=4, k_w=4, name='d_sn_1_conv0')))
                # sn_1_1 = lrelu(self.d_sn_1_bn1(conv2d(sn_1_0, 64, k_w=4, k_h=4, name='d_sn_1_conv1')))
                # sn_1_2 = lrelu(linear(tf.reshape(sn_1_1, [self.batch_size, -1]), 256, name='d_sn1_linear0'))
                # # sub net 2
                # sn_2_0 = lrelu(self.d_sn_2_bn0(conv2d(img2, 32, k_h=4, k_w=4, name='d_sn_2_conv0')))
                # sn_2_1 = lrelu(self.d_sn_2_bn1(conv2d(sn_2_0, 64, k_w=4, k_h=4, name='d_sn_2_conv1')))
                # sn_2_2 = lrelu(linear(tf.reshape(sn_2_1, [self.batch_size, -1]), 256, name='d_sn2_linear0'))
                # # compose sub output
                # d0 = tf.reshape(tf.pack([sn_0_2, sn_1_2, sn_2_2]), [self.batch_size, -1])
                # discriminator_template = linear(d0, 1, name='d_real_prob')
                # encoder_template = linear(lrelu(linear(d0, 128)), self.reg_latent_dist.dist_flat_dim, name='d_noise_and_code')
                #
                # d = tf.nn.sigmoid(discriminator_template[:, 0])
                # reg_dist_flat = tf.identity(encoder_template, name='reg_dist_flat')
                # reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
                # return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, d0
        elif self.network_type == 'celeba':
            with tf.variable_scope('d_net', reuse=reuse):
                img = tf.reshape(x_var, shape=[-1]+list(self.image_shape))
                h0 = lrelu(conv2d(img, output_dim=64, k_w=4, k_h=4, name='d_face_conv0'))
                h1 = lrelu(self.d_face_bn0(conv2d(h0, output_dim=128, k_w=4, k_h=4, name='d_face_conv1')))
                h2 = lrelu(self.d_face_bn1(conv2d(h1, output_dim=256, k_w=4, k_h=4, name='d_face_conv2')))
                h3 = lrelu(self.d_face_bn2(conv2d(h2, output_dim=512, k_h=4, k_w=4, name='d_face_conv3')))
                h4 = tf.reshape(h3, [self.batch_size, -1])
                discriminator_template = linear(h4, 1, name='d_face_real_prob')
                encoder_template = linear(lrelu(linear(h4, 126, name='d_face_linear0')), self.reg_latent_dist.dist_flat_dim,
                                          name='d_face_noise_code')

                d_out = tf.identity(discriminator_template, name='d_out')
                d = tf.nn.sigmoid(d_out[:, 0])
                reg_dist_flat = tf.identity(encoder_template, name='reg_dist_flat')
                reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
                return d, self.reg_latent_dist.sample(reg_dist_info)
                #, reg_dist_info, reg_dist_flat
        elif self.network_type == 'celeba_vgg':
            with tf.variable_scope('d_net', reuse=reuse):
                img = tf.reshape(x_var, shape=[-1]+list(self.image_shape))
                vgg_fea, _ = vgg_net(self.vgg_model, img)
                h0 = lrelu(conv2d(vgg_fea['relu2_1'], output_dim=16, k_w=3, k_h=3, name='d_face_conv0'))
                h1 = lrelu(linear(tf.reshape(h0, [self.batch_size, -1]), output_size=256, name='d_face_linear0'))
                content_discriminator = tf.nn.sigmoid(linear(h1, 1, name='d_face_content_linear0'))
                h2 = lrelu(conv2d(vgg_fea['relu5_1'], output_dim=16, k_h=3, k_w=3, name='d_face_conv1'))
                h3 = lrelu(linear(tf.reshape(h2, [self.batch_size, -1]), output_size=256, name='d_face_linear1'))
                style_discriminatror = tf.nn.sigmoid(linear(h3, 1, name='d_face_style_linear1'))
                return content_discriminator, style_discriminatror

        else:
            raise NotImplementedError

    def generate(self, z_var, reuse=None):
        if self.network_type == 'mnist':
            with tf.variable_scope('g_net', reuse=reuse):
                return
                # noise_code = tf.reshape(z_var, [self.batch_size, self.latent_dist.dim])
                # #h0 = lrelu(self.ge_bn0(tf.reshape(linear(noise_code, 1024), ), name='ge_linear0')
                # h0 = lrelu(linear(noise_code, 1024, name='ge_linear0'))
                # h1 = lrelu(self.ge_bn1(tf.reshape(linear(h0, 7*7*128, name='ge_linear1'), [self.batch_size,7,7,128])))
                # h2 = lrelu(self.ge_bn2(deconv2d(h1, output_shape=[self.batch_size,14, 14, 64], k_h=4,k_w=4, name='ge_deconv0')))
                # x_dist_flat = tf.reshape(deconv2d(h2, output_shape=[self.batch_size, 28, 28, 1], name='ge_deconv1'),
                #                          shape=[self.batch_size, -1])
                # x_dist_info = self.output_dist.activate_dist(x_dist_flat)
                # return self.output_dist.sample(x_dist_info), x_dist_info
        elif self.network_type == 'de_mnist':
            with tf.variable_scope('g_net', reuse=reuse):
                return
                # noise_code = tf.reshape(z_var, [self.batch_size, self.latent_dist.dim])
                # h0 = lrelu(linear(noise_code, 1024, name='ge_sn_linear0'))
                # # sub net 0
                # sn_0_0 = lrelu(linear(h0, 7*7*128, name='ge_sn_0_linear0'))
                # sn_0_1 = lrelu(self.ge_sn_0_bn0(tf.reshape(sn_0_0, [self.batch_size,7,7,128])))
                # sn_0_2 = lrelu(self.ge_sn_0_bn1(deconv2d(sn_0_1, output_shape=[self.batch_size,14, 14, 64], k_h=4,k_w=4, name='ge_sn_0_deconv0')))
                # sn_0 = deconv2d(sn_0_2, output_shape=[self.batch_size, 28, 28, 1], name='ge_sn_0_deconv1')
                #
                # # sub net 1
                # sn_1_0 = lrelu(linear(h0, 7 * 7 * 128, name='ge_sn_1_linear0'))
                # sn_1_1 = lrelu(self.ge_sn_1_bn0(tf.reshape(sn_1_0, [self.batch_size, 7, 7, 128])))
                # sn_1_2 = lrelu(self.ge_sn_1_bn1(
                #     deconv2d(sn_1_1, output_shape=[self.batch_size, 14, 14, 64], k_h=4, k_w=4, name='ge_sn_1_deconv0')))
                # sn_1 = deconv2d(sn_1_2, output_shape=[self.batch_size, 28, 28, 1], name='ge_sn_1_deconv1')
                #
                # # sub net 2
                # sn_2_0 = lrelu(linear(h0, 7 * 7 * 128, name='ge_sn_2_linear0'))
                # sn_2_1 = lrelu(self.ge_sn_2_bn0(tf.reshape(sn_2_0, [self.batch_size, 7, 7, 128])))
                # sn_2_2 = lrelu(self.ge_sn_2_bn1(
                #     deconv2d(sn_2_1, output_shape=[self.batch_size, 14, 14, 64], k_h=4, k_w=4, name='ge_sn_2_deconv0')))
                # sn_2 = deconv2d(sn_2_2, output_shape=[self.batch_size, 28, 28, 1], name='ge_sn_2_deconv1')
                #
                # # compose subnets
                # inv_theta0 = inverse_theta(self.theta0_list)
                # inv_theta1 = inverse_theta(self.theta1_list)
                # inv_theta2 = inverse_theta(self.theta2_list)
                #
                # output_size = [28,28]
                # sub_out0 = inverse_transformer(sn_0, inv_theta0, output_size, self.batch_size)
                # sub_out1 = inverse_transformer(sn_1, inv_theta1, output_size, self.batch_size)
                # sub_out2 = inverse_transformer(sn_2, inv_theta2, output_size, self.batch_size)
                # ge_out = tf.add_n([sub_out0, sub_out1, sub_out2])
                # return ge_out, self.output_dist.activate_dist(ge_out)
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
                                    int(ceil(self.image_shape[1]/8.)), 64], k_h=4, k_w=4,
                                    name='ge_face_deconv1')))
                h3 = lrelu(self.ge_face_bn2(deconv2d(h2, output_shape=[self.batch_size, int(ceil(self.image_shape[0]/4.)),
                                    int(ceil(self.image_shape[1]/4.)), 32], k_h=4, k_w=4,
                                    name='ge_face_deconv2')))
                h4 = lrelu(self.ge_face_bn3(deconv2d(h3, output_shape=[self.batch_size, int(ceil(self.image_shape[0]/2.)),
                                                      int(ceil(self.image_shape[1]/2.)), 16], k_h=4, k_w=4,
                                    name='ge_face_deconv3')))
                x_dist_flat = tf.nn.tanh(self.ge_face_bn4(deconv2d(h4, output_shape=[self.batch_size] + self.image_shape, k_w=4, k_h=4,
                                    name='ge_face_deconv4')))
                x_dist_info = self.output_dist.activate_dist(x_dist_flat)
                return x_dist_flat, x_dist_info
        elif self.network_type == 'celeba_vgg':
            with tf.variable_scope('g_net', reuse=reuse):
                noise_code = tf.reshape(z_var, [self.batch_size, self.latent_dist.dim])
                h0 = lrelu(
                    linear(noise_code, int(ceil(self.image_shape[0] / 32.)) * int(ceil(self.image_shape[1] / 32.)) * 256,
                           name='ge_face_linear0'))
                h1 = lrelu(self.ge_face_bn0(deconv2d(tf.reshape(h0, shape=[self.batch_size,
                                                                           int(ceil(self.image_shape[0] / 32.)),
                                                                           int(ceil(self.image_shape[1] / 32.)),
                                                                           256]),
                                                     output_shape=[self.batch_size,
                                                                   int(ceil(self.image_shape[0] / 16.)),
                                                                   int(ceil(self.image_shape[1] / 16.)), 128], k_h=4, k_w=4,
                                                     name='ge_face_deconv0')))
                h2 = lrelu(self.ge_face_bn1(deconv2d(h1, output_shape=[self.batch_size, int(ceil(self.image_shape[0] / 8.)),
                                                                       int(ceil(self.image_shape[1] / 8.)), 64], k_h=4,
                                                     k_w=4,
                                                     name='ge_face_deconv1')))
                h3 = lrelu(self.ge_face_bn2(deconv2d(h2, output_shape=[self.batch_size, int(ceil(self.image_shape[0] / 4.)),
                                                                       int(ceil(self.image_shape[1] / 4.)), 32], k_h=4,
                                                     k_w=4,
                                                     name='ge_face_deconv2')))
                h4 = lrelu(self.ge_face_bn3(deconv2d(h3, output_shape=[self.batch_size, int(ceil(self.image_shape[0] / 2.)),
                                                                       int(ceil(self.image_shape[1] / 2.)), 16], k_h=4,
                                                     k_w=4,
                                                     name='ge_face_deconv3')))
                x_dist_flat = tf.nn.tanh(
                    self.ge_face_bn4(deconv2d(h4, output_shape=[self.batch_size] + self.image_shape, k_w=4, k_h=4,
                                              name='ge_face_deconv4')))
                x_dist_info = self.output_dist.activate_dist(x_dist_flat)
                return x_dist_flat, x_dist_info
        else:
            raise NotImplementedError

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




