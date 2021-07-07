import os
import disentanglement_lib.utils.hyperparams as h
import argparse

from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised.model import Regularizer

os.environ['WANDB_PROJECT'] = 'uncertainty'
os.environ['WANDB_TAGS'] = 'supervision'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gin
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.utils
import wandb
from disentanglement_lib.methods.unsupervised.train import Train

from disentanglement_lib.methods.shared.architectures import View
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F


@gin.configurable("supervision")
class Supervision(Regularizer):
    """AnnealedVAE model."""

    def __init__(self, order=gin.REQUIRED, beta =1, steps=1000):
        super().__init__()
        self.order = order
        self.beta = beta
        self.steps = steps

        mean = torch.tensor([1.5, 11.5, 91.0])
        std = torch.tensor([1.118033988749895, 6.922186552431729, 52.82676089508675])
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, data_batch, model, kl, z_mean, z_logvar, z_sampled):

        stage = (model.global_step // self.steps)
        if stage < len(self.order):
            _, labels = data_batch
            factors = (labels-self.mean) / self.std
            l_sup=F.mse_loss(z_mean[:,stage], factors[:,self.order[stage]])
            model.summary['supervision'] = l_sup.item()
            return l_sup * self.beta
        else:
            return 0

# gin.enter_interactive_mode()

gin_bindings = ['vae.beta=6',
                'supervision.beta=10',
                'model.regularizers=[@vae,@supervision]',
                'model.num_latent = 10',
                'model.encoder_fn = @conv_encoder',
                'model.decoder_fn = @deconv_decoder',
                'reconstruction_loss.loss_fn = @bernoulli_loss',
                "train.dataset='cars3d'",
                "dataset.name='cars3d'",
                'train.training_steps=5000',
                'train.eval_callbacks=[@eval_mig]',
                'train.random_seed=99',
                'train.batch_size=256',
                'eval_mig.evaluation_steps=50',
                'discretizer.discretizer_fn = @histogram_discretizer',
                'discretizer.num_bins = 20',
                'mig.num_train = 10000'
                ]

from itertools import permutations
for order in permutations(range(3)):
    order_binding=[f'supervision.order={order[:2]}']
    train.train_with_gin('tmp',True,None,gin_bindings=gin_bindings+order_binding)

