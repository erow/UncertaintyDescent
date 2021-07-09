import os
import disentanglement_lib.utils.hyperparams as h
import argparse
import numpy as np
from disentanglement_lib.methods.unsupervised.train import Train, pl
from disentanglement_lib.methods.unsupervised.model import Regularizer
from pytorch_lightning.loggers import WandbLogger

os.environ['WANDB_ENTITY'] = 'dlib'
os.environ['WANDB_PROJECT'] = 'uncertainty'
os.environ['WANDB_TAGS'] = 'supervision'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gin
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data

from torch import nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

data = get_named_ground_truth_data(args.dataset)


@gin.configurable("supervision")
class Supervision(Regularizer):
    """AnnealedVAE model."""

    def __init__(self, order=gin.REQUIRED, beta=1, steps=1000):
        super().__init__()
        self.order = order
        self.beta = beta
        self.steps = steps

        mean = torch.tensor([np.mean(range(i)) for i in data.factors_num_values])
        std = torch.tensor([np.std(range(i)) for i in data.factors_num_values])
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        print(self.mean,self.std)

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):

        stage = (model.global_step // self.steps)
        if stage < len(self.order):

            _, labels = data_batch
            factors = (labels - self.mean) / self.std
            l_sup = F.mse_loss(z_sampled[:, stage], factors[:, self.order[stage]])
            model.summary['supervision'] = l_sup.item()
            return l_sup * self.beta
        else:
            return 0


gin_bindings = ['vae.beta=6',
                'supervision.beta=10',
                'model.regularizers=[@vae, @supervision]',
                'model.num_latent = 10',
                'model.encoder_fn = @conv_encoder',
                'model.decoder_fn = @deconv_decoder',
                'reconstruction_loss.loss_fn = @bernoulli_loss',
                f"train.dataset='{args.dataset}'",
                f"dataset.name='{args.dataset}'",
                'train.training_steps=5000',
                'train.eval_callbacks=[@eval_mig]',
                f'train.random_seed={args.seed}',
                'train.batch_size=256',
                'eval_mig.evaluation_steps=50',
                'discretizer.discretizer_fn = @histogram_discretizer',
                'discretizer.num_bins = 20',
                'mig.num_train = 10000'
                ]

from itertools import permutations

for order in permutations(range(data.num_factors)):
    order_binding = [f'supervision.order={order[:data.num_factors - 1]}']
    try:
        logger = WandbLogger()
        pl_model = Train()
        trainer = pl.Trainer(logger,
                             progress_bar_refresh_rate=500,  # disable progress bar
                             max_steps=pl_model.training_steps,
                             checkpoint_callback=False,
                             gpus=1,)
        trainer.fit(pl_model)
        pl_model.save_model('model.pt', 'tmp')
    except:
        pass
    finally:
        wandb.join()
