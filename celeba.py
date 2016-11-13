from __future__ import print_function
from __future__ import absolute_import
from distribution import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from datasets import CelebA
from  model import InfoGAN
from InfoTrainer import InfoGANTrainer
from utils import mkdir_p, mnist_hist
import dateutil.tz
import datetime
if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%d_%H_%M_%S')

    root_log_dir = "logs/CelebA"
    root_checkpoint_dir = "ckt/CelebA"
    batch_size = 128
    updates_per_epoch = 200
    max_epoch = 50

    exp_name = "CelebA_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    dataset = CelebA()

    latent_spec = [
        #(Uniform(1), False),
        # (Categorical(10), True),
        # (Categorical(10), True),
        (Uniform(100, fix_std=False), True),
    ]

    model = InfoGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="celeba",
        vgg_path='../imagenet-vgg-verydeep-19.mat'
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        snapshot_interval=5000,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-4,
        discriminator_learning_rate=1e-4,
        vae_learning_rate=1e-3,
        decay_value=1e-6,
        method_type='vae'
    )

    algo.train()
    #algo.eval_generated_images(save_path='Celeba_vae_small_kernel.ckpt', best_num=50, iterations=100)
    tmp = 123




