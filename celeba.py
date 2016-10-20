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
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/CelebA"
    root_checkpoint_dir = "ckt/CelebA"
    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 50

    exp_name = "CelebA_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    dataset = CelebA()

    latent_spec = [
        (Uniform(128), False),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Categorical(10), True),
        (Uniform(20), False),

    ]

    model = InfoGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="celeba",
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
        snapshot_interval=-1,
        info_reg_coeff=1.0,
        generator_learning_rate=8e-3,
        discriminator_learning_rate=9e-4,
        reload=False,
        save_path='Celeba_half_unif_8e-3_4e-4_crop.ckpt'
    )

    algo.train()
    #algo.classify(test_images=None)
    #test_set = dataset.test.images[0:9984]
    #test_set_labels = dataset.test.labels[0:9984]
    #
    # result = algo.classify(test_set)
    # for label in range(10):
    #     ids = np.where(test_set_labels == label)
    #     mnist_hist(np.array(result)[ids[0]], title=label )
    #manifold_ene, real_data_ene = algo.evaluate_energy(images=test_set, epoches=1)

    tmp = 123




