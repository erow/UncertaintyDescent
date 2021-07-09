from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from disentanglement_lib.data.ground_truth import named_data
from uncertainty_estimation import *

if __name__ == '__main__':
    ds = named_data.get_named_ground_truth_data("cars3d")

    # random seed
    k = 5
    num_factor = ds.num_factors
    fig1,ax1 = plt.subplots()
    fig,axes = plt.subplots(k,num_factor,figsize=(7*num_factor,7*k),dpi=100)
    for j in range(k):
        Hds = compute_uncertainty(ds, GaussianH,range(num_factor))
        t=(Hds).sum(1).sum(1)
        ax1.plot(t,label=j)
        for i in range(num_factor):
            ax = axes[j,i]
            cm=ax.imshow(Hds[i])
            plt.colorbar(cm,ax=ax)
    ax1.legend()

    # order
    # k = 4
    # num_factor = ds.num_factors
    # fig1,ax1 = plt.subplots()
    # fig,axes = plt.subplots(k,num_factor,figsize=(7*num_factor,7*k),dpi=100)
    # for j in range(k):
    #     order = np.random.permutation(range(num_factor))
    #     Hds = Hx(ds,GaussianH, order, 10000)
    #     t=(Hds).sum(1).sum(1)
    #     ax1.plot(t,label=order)
    #     for i in range(num_factor):
    #         ax = axes[j,i]
    #         cm=ax.imshow(Hds[i])
    #         plt.colorbar(cm,ax=ax)
    #         ax.set_title(order[:i])
    # ax1.legend()

