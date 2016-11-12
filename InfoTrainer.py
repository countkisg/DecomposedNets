from model import InfoGAN
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from distribution import Bernoulli, Gaussian, Categorical
from ops import binary_crossentropy
import scipy.io
import sys
import scipy.misc
from math import ceil
from utils import mkdir_p
import dateutil.tz
import datetime
import os
TINY = 1e-8
class InfoGANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 dataset=None,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=-1,
                 info_reg_coeff=1.0,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 save_path=None,
                 method_type=None
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = max_epoch*updates_per_epoch if -1 == snapshot_interval else snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.trainer_dict = dict()

        self.input_tensor = None
        self.test_tensor = None # Only used after trained
        self.log_vars = []
        self.reload = reload
        self.save_path = save_path
        self.d_loss = None
        self.g_loss = None
        self.method_type = method_type

    def init_vae_opt(self):
        with tf.Session() as sess:
            self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])

            real_d, real_code_info = self.model.discriminate(input_tensor, reuse=False)
            fake_x, _ = self.model.generate(self.model.reg_latent_dist.sample(real_code_info), reuse=False)

            vae_loss_recons = self.vae_loss_recons(input_tensor, fake_x)
            vae_loss_kl = self.vae_loss_kl(real_code_info)

            self.log_vars.append(("vae_loss_recons", vae_loss_recons))
            self.log_vars.append(("vae_loss_kl", vae_loss_kl))

            all_vars = tf.trainable_variables()
            self.d_vars = [var for var in all_vars if var.name.startswith('d_')]
            self.g_vars = [var for var in all_vars if var.name.startswith('g_')]

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            for k, v in self.log_vars:
                tf.scalar_summary(k, v)

            vae_loss = vae_loss_recons + vae_loss_kl

            vae_trainer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(vae_loss, var_list=self.d_vars+self.g_vars)
            self.trainer_dict['vae_trainer'] = vae_trainer

    def init_gan_opt(self):
        with tf.Session() as sess:
            self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])
            self.z_var = z_var = self.model.latent_dist.sample_prior(self.batch_size)

            real_d, real_code_info = self.model.discriminate(input_tensor, reuse=False)
            fake_x, _ = self.model.generate(self.z_var, reuse=False)
            fake_d, fake_code_info = self.model.discriminate(fake_x, reuse=True)

            discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) +
                                                  tf.log(1. - fake_d + TINY))
            generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))

            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))

            all_vars = tf.trainable_variables()
            self.d_vars = [var for var in all_vars if var.name.startswith('d_')]
            self.g_vars = [var for var in all_vars if var.name.startswith('g_')]

            # thetas = [var for var in all_vars if var.name.startswith('theta')]

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))
            for k, v in self.log_vars:
                tf.scalar_summary(k, v)

            self.global_step = tf.Variable(0, trainable=False)
            d_learning_rate = tf.train.exponential_decay(self.discriminator_learning_rate, self.global_step,
                                                       500, 0.8, staircase=True)
            discriminator_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=0.5)
            discriminator_trainer = discriminator_optimizer.minimize(self.d_loss, var_list=self.d_vars)

            g_learning_rate = tf.train.exponential_decay(self.generator_learning_rate, self.global_step,
                                                       500, 0.8, staircase=True)
            generator_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=0.5)
            generator_trainer = generator_optimizer.minimize(self.g_loss, var_list=self.g_vars)
            self.trainer_dict['d_trainer'] = discriminator_trainer
            self.trainer_dict['g_trainer'] = generator_trainer

    def vae_loss_recons(self, real_im, fake_im):
        epsilon = 1e-8
        fake_im = tf.reshape(fake_im, [self.batch_size, -1])
        real_im = (real_im+1.)/2.
        fake_im = (fake_im+1.)/2.
        #recons_error = tf.reduce_sum(binary_crossentropy(fake_im, real_im))
        recons_error = tf.reduce_sum((tf.square(real_im-fake_im)))
        # recons_error = tf.reduce_sum(-real_im * tf.log(fake_im + epsilon) -
        #             (1.0 - real_im) * tf.log(1.0 - fake_im + epsilon))
        return recons_error

    def vae_loss_kl(self, code_info):
        kl_loss = tf.reduce_sum( -1. - 2.*tf.log(code_info['id_0_stddev'])
                                         + tf.square(code_info['id_0_mean'])
                                         + tf.square(code_info['id_0_stddev']))
        return kl_loss

    def visualize_all_factors(self):
        with tf.Session():
            fixed_noncat = np.concatenate([
                np.tile(
                    self.model.nonreg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)
            fixed_cat = np.concatenate([
                np.tile(
                    self.model.reg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)

        offset = 0
        for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
            if isinstance(dist, Gaussian):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                c_vals = []
                for idx in xrange(10):
                    c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                c_vals.extend([0.] * (self.batch_size - 100))
                vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+1] = vary_cat
                offset += 1
            elif isinstance(dist, Categorical):
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([idx] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([int(idx / 5)] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                # import ipdb; ipdb.set_trace()
                offset += dist.dim
            else:
                raise NotImplementedError
            z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))

            _, x_dist_info = self.model.generate(z_var, reuse=True)

            # just take the mean image
            if isinstance(self.model.output_dist, Bernoulli):
                img_var = x_dist_info["p"]
            elif isinstance(self.model.output_dist, Gaussian):
                img_var = x_dist_info["mean"]
            else:
                raise NotImplementedError
            img_var = self.dataset.inverse_transform(img_var)
            rows = 10
            img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
            img_var = img_var[:rows * rows, :, :, :]
            imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
            stacked_img = []
            for row in xrange(rows):
                row_img = []
                for col in xrange(rows):
                    row_img.append(imgs[row, col, :, :, :])
                stacked_img.append(tf.concat(1, row_img))
            imgs = tf.concat(0, stacked_img)
            imgs = tf.expand_dims(imgs, 0)
            tf.image_summary("image_%d_%s" % (dist_idx, dist.__class__.__name__), imgs)

    def train(self):
        if 'vae' == self.method_type.lower():
            self.init_vae_opt()
        elif 'gan' == self.method_type.lower():
            self.init_gan_opt()
        elif 'vgan' == self.method_type.lower():
            return
        else:
            raise NotImplementedError

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            counter = 0

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, _ = self.dataset.train.next_batch(self.batch_size)
                    feed_dict = {self.input_tensor: x}

                    # run different optimization objectives
                    for k, v in self.trainer_dict.iteritems():
                        log_vals = sess.run([v] + log_vars, feed_dict)[1:]
                    # End optimization
                    all_log_vals.append(log_vals)
                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x, _ = self.dataset.train.next_batch(self.batch_size)

                summary_str = sess.run(summary_op, {self.input_tensor: x})
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")
        return

    def eval_generated_images(self, best_num=10, iterations=10, ):
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%m_%d_%H_%M')

        root_log_dir = "selected_result/"

        exp_name = "%s_%s_%s" % (self.method_type, self.model.nework_type, timestamp)

        log_dir = os.path.join(root_log_dir, exp_name)
        mkdir_p(log_dir)

        with tf.Session() as sess:
            if 'vae' == self.method_type.lower():
                self.init_vae_opt()
            elif 'gan' == self.method_type.lower():
                self.init_gan_opt()
            elif 'vgan' == self.method_type.lower():
                return
            else:
                raise NotImplementedError

            init = tf.initialize_all_variables()
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            best_loss = np.empty(shape=(0) ,dtype=np.float32)
            best_result = np.zeros(shape=[best_num]+self.dataset.image_shape)
            for i in range(iterations):
                x, _ = self.dataset.train.next_batch(self.batch_size)
                feed_dict = {self.input_tensor: x}
                generated_img = None
                loss = None
                if 'vae' == self.method_type.lower():
                    real_d, real_code_info = self.model.discriminate(self.input_tensor, reuse=True)
                    fake_x, _ = self.model.generate(self.model.reg_latent_dist.sample(real_code_info), reuse=True)

                    vae_loss_recons = self.vae_loss_recons(self.input_tensor, fake_x)
                    vae_loss_kl = self.vae_loss_kl(real_code_info)
                    generated_img, loss = sess.run([fake_x, vae_loss_kl+vae_loss_recons], feed_dict=feed_dict)
                elif 'gan' == self.method_type.lower():
                    z_var = tf.placeholder(tf.float32, [self.batch_size, self.model.latent_dist.dim])
                    fake_x, _ = self.model.generate(z_var, reuse=True)
                    fake_d, _ = self.model.discriminate(fake_x, reuse=True)
                    feed_dict = {z_var:np.random.uniform(-1., 1., size=(self.batch_size, self.model.latent_dist.dim))}
                    generated_img, loss = sess.run([fake_x, fake_d], feed_dict=feed_dict)
                elif 'vgan' == self.method_type.lower():
                    return
                else:
                    raise NotImplementedError
                best_loss = np.concatenate((loss, best_loss))
                index = np.argmin(best_loss)
                best_loss = best_loss[index[0:best_num]]
                best_result = np.concatenate((generated_img, best_result))[index[0:best_num]]
            # save images
            for i in range(best_num):
                path = os.path.join(log_dir, '%03d.jpg' % i)
                scipy.misc.imsave(path, best_result[i])

    def save_decoded_images(self, new_dir='decoded_'):
        with tf.Session() as sess:
            if 'vae' == self.method_type.lower():
                self.init_vae_opt()
            elif 'gan' == self.method_type.lower():
                self.init_gan_opt()
            elif 'vgan' == self.method_type.lower():
                return
            else:
                raise NotImplementedError

            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            test_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])
            _, real_code_info = self.model.discriminate(test_tensor, reuse=True)
            fake_images, _ = self.model.generate(self.model.reg_latent_dist.sample(real_code_info), reuse=True)
            iterations = int(ceil(self.dataset.train.num_examples / float(self.batch_size)))
            for i in range(iterations):
                im, paths = self.dataset.train.next_sequencial_batch(self.batch_size)
                decoded_im = sess.run(fake_images,
                                      feed_dict={
                                          test_tensor: im
                                      })

                decoded_im = (decoded_im+1.)/2.
                for j in range(self.batch_size):
                    scipy.misc.imsave(new_dir+paths[j], decoded_im[j])