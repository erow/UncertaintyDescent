import os
import disentanglement_lib.utils.hyperparams as h
import argparse
import numpy as np
from disentanglement_lib.methods.unsupervised.train import Train, pl
from disentanglement_lib.methods.unsupervised.model import Regularizer
from pytorch_lightning.loggers import WandbLogger

os.environ['WANDB_ENTITY'] = 'dlib'
os.environ['WANDB_PROJECT'] = 'uncertainty'
os.environ['WANDB_TAGS'] = 'supervision_factor'
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

    def __init__(self, factor=gin.REQUIRED, beta=1, steps=500):
        super().__init__()
        self.factor = factor
        self.beta = beta
        self.steps = steps

        mean = torch.tensor([np.mean(range(i)) for i in data.factors_num_values]).float()
        std = torch.tensor([np.std(range(i)) for i in data.factors_num_values]).float()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        print(self.mean,self.std)

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):
        if model.global_step == self.steps:
            pl_model.save_model('model_500.pt','tmp')

        if model.global_step < self.steps:
            _, labels = data_batch
            factors = (labels - self.mean) / self.std
            l_sup = F.mse_loss(z_sampled[:, 0], factors[:, self.factor])
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
                f'train.random_seed={args.seed}',
                'train.training_steps=1000',
                'train.batch_size=256',
                'train.eval_callbacks=[@eval_mig,@eval_cmi,@eval_decomposition]',
                'eval_mig.evaluation_steps=100',
                'eval_cmi.evaluation_steps=100',
                'eval_decomposition.evaluation_steps=100',
                'supervision.steps=500',
                'discretizer.discretizer_fn = @histogram_discretizer',
                'discretizer.num_bins = 20',
                'mig.num_train = 10000'
                ]

from itertools import permutations

for factor in (range(data.num_factors)):
    order_binding = [f'supervision.factor={factor}']
    try:
        gin.parse_config(gin_bindings + order_binding)
        logger = WandbLogger()
        pl_model = Train(dir='tmp')
        trainer = pl.Trainer(logger,
                             progress_bar_refresh_rate=500,  # disable progress bar
                             max_steps=pl_model.training_steps,
                             checkpoint_callback=False,
                             gpus=1,)
        trainer.fit(pl_model)
        pl_model.save_model('model.pt', 'tmp')
    except Exception as e:
        print(e)
    finally:
        wandb.join()
