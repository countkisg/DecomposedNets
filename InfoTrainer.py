from model import InfoGAN
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from distribution import Bernoulli, Gaussian, Categorical
from ops import binary_crossentropy
import scipy.io
import sys
import math
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
                 reload=False,
                 save_path=None,
                 trans_loss_coeff=0.2,
                 style_loss_coeff=1
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
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.trans_trainer = None
        self.input_tensor = None
        self.test_tensor = None # Only used after trained
        self.log_vars = []
        self.reload = reload
        self.save_path = save_path
        self.trans_loss_coeff = trans_loss_coeff
        self.style_loss_coeff = style_loss_coeff
        self.d_loss = None
        self.g_loss = None
        ################
        self.subnets = 3 # At least larger than 1


    def init_opt(self):
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])

        with tf.Session() as sess:
            self.z_var = z_var = self.model.latent_dist.sample_prior(self.batch_size)

            real_d, real_code_info = self.model.discriminate(input_tensor, reuse=False)

            fake_x, _ = self.model.generate(self.model.reg_latent_dist.sample(real_code_info), reuse=False)
            fake_d, fake_code_info = self.model.discriminate(fake_x, reuse=True)

            reg_z = self.model.reg_z(z_var)

            discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) +
                                                  tf.log(1. - fake_d + TINY))
            generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY) )

            vae_loss = self.vae_loss(input_tensor, fake_x, real_code_info)

            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))
            self.log_vars.append(("vae_loss", vae_loss))

            # compute for transformer loss
            # trans_mean, trans_var = tf.nn.moments(region_feature, axes=[0])
            # inside_loss = tf.reduce_mean(trans_var)
            # trans_mean = tf.reshape(trans_mean, [self.subnets, -1])
            # outside_loss = tf.constant(0.)
            # for i in range(self.subnets+1):
            #     outside_loss = outside_loss + \
            #                    binary_crossentropy(trans_mean[:, i%self.subnets], trans_mean[:, (i+1)%self.subnets])
            # outside_loss = - tf.log(tf.reduce_mean(outside_loss))
            # trans_loss = inside_loss + outside_loss*self.trans_loss_coeff
            # discriminator_loss += trans_loss

            # self.log_vars.append(("transformer_loss", trans_loss))
            # self.log_vars.append(("inside_loss", inside_loss))
            # self.log_vars.append(("outside_loss", outside_loss))

            # compute for discrete and continuous codes separately
            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)
            #test_only_entropy = tf.constant(0.)
            # discrete:
            # if len(self.model.reg_disc_latent_dist.dists) > 0:
            #     disc_reg_z = self.model.disc_reg_z(reg_z)
            #     disc_reg_dist_info = self.model.disc_reg_dist_info(fake_reg_z_dist_info)
            #     disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
            #     disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
            #     disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
            #     disc_ent = tf.reduce_mean(-disc_log_q_c)
            #     disc_mi_est = disc_ent - disc_cross_ent
            #     mi_est += disc_mi_est
            #     cross_ent += disc_cross_ent
            #     self.log_vars.append(("MI_disc", disc_mi_est))
            #     self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
            #     discriminator_loss -= self.info_reg_coeff * disc_mi_est
            #     generator_loss -= self.info_reg_coeff * disc_mi_est
            #     test_only_entropy -= self.info_reg_coeff * disc_mi_est
            #
            # if len(self.model.reg_cont_latent_dist.dists) > 0:
            #     cont_reg_z = self.model.cont_reg_z(reg_z)
            #     cont_reg_dist_info = self.model.cont_reg_dist_info(fake_reg_z_dist_info)
            #     cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
            #     cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
            #     cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
            #     cont_ent = tf.reduce_mean(-cont_log_q_c)
            #     cont_mi_est = cont_ent - cont_cross_ent
            #     mi_est += cont_mi_est
            #     cross_ent += cont_cross_ent
            #     self.log_vars.append(("MI_cont", cont_mi_est))
            #     self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
            #     discriminator_loss -= self.info_reg_coeff * cont_mi_est
            #     generator_loss -= self.info_reg_coeff * cont_mi_est
            #     test_only_entropy -= self.info_reg_coeff * cont_mi_est

            # for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
            #     if "stddev" in dist_info:
            #         self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
            #         self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

            self.log_vars.append(("MI", mi_est))
            self.log_vars.append(("CrossEnt", cross_ent))

            all_vars = tf.trainable_variables()
            self.d_vars = [var for var in all_vars if var.name.startswith('d_')]
            self.g_vars = [var for var in all_vars if var.name.startswith('g_')]

            #thetas = [var for var in all_vars if var.name.startswith('theta')]

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))
            for k, v in self.log_vars:
                tf.scalar_summary(k, v)

            self.d_loss = discriminator_loss
            self.g_loss = generator_loss
            self.vae_loss = vae_loss

            self.global_step = tf.Variable(0, trainable=False)
            d_learning_rate = tf.train.exponential_decay(self.discriminator_learning_rate, self.global_step,
                                                       500, 0.8, staircase=True)
            discriminator_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=0.5)
            self.discriminator_trainer = discriminator_optimizer.minimize(self.d_loss, var_list=self.d_vars)

            g_learning_rate = tf.train.exponential_decay(self.generator_learning_rate, self.global_step,
                                                       500, 0.8, staircase=True)
            generator_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=0.5)
            self.generator_trainer = generator_optimizer.minimize(self.g_loss, var_list=self.g_vars)

            self.vae_trainer = tf.train.RMSPropOptimizer(learning_rate=4e-4).minimize(self.vae_loss, var_list=self.d_vars+self.g_vars)

    def vae_loss(self, real_im, fake_im, code_info):
        fake_im = tf.reshape(fake_im, [self.batch_size, -1])
        real_im = (real_im+1.)/2.
        fake_im = (fake_im+1.)/2.
        recons_error = tf.reduce_mean(binary_crossentropy(fake_im, real_im))
        kl_loss = - 0.5 * tf.reduce_mean(1. + 0. - tf.square(code_info['id_0_mean'] - 1.))
        return recons_error+kl_loss

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

        self.init_opt()

        init = tf.initialize_all_variables()
        initial_vars = set(tf.all_variables())
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
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, _ = self.dataset.train.next_batch(self.batch_size)
                    feed_dict = {self.input_tensor: x}

                    assign_op = self.global_step.assign(epoch*self.updates_per_epoch + i)
                    sess.run(assign_op)
                    log_vals = sess.run([self.vae_trainer] + log_vars, feed_dict)[1:]
                    #sess.run(self.generator_trainer, feed_dict)
                    #print sess.run(self.z_var)
                    all_log_vals.append(log_vals)
                    counter += 1

                    #g_loss, d_loss = sess.run([self.g_loss, self.d_loss], feed_dict)
                    #print g_loss, d_loss
                    #if g_loss > d_loss / 2:
                    #    sess.run(self.generator_trainer, feed_dict)

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

    def classify(self, test_images):
        with tf.Session() as sess:
            self.init_opt()

            init = tf.initialize_all_variables()
            sess.run(init)

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            self.test_tensor = test_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])
            d, sample, reg_dist_info, reg_dist_flat = self.model.discriminate(test_tensor, reuse=True)
            length, _ = test_images.shape
            result = []
            for iteration in range(length/self.batch_size):
                from_idx = iteration*self.batch_size
                to_idx = from_idx+self.batch_size
                output = sess.run(reg_dist_info, feed_dict={test_tensor: test_images[from_idx:to_idx, :]})
                result.extend(np.argmax(output['id_0_prob'], axis=1))
            return result

