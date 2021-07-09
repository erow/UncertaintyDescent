from __future__ import print_function, division

from itertools import permutations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from disentanglement_lib.data.ground_truth import named_data
from uncertainty_estimation import *

if __name__ == '__main__':
    for seed in range(3):
        for ds_name in ['cars3d', 'dsprites_full', 'smallonorb']:
            ds = named_data.get_named_ground_truth_data(ds_name)
            for i,order in permutations(range(ds.num_factors)):
                entropy_func =  BernoulliH if ds_name == "dsprites_full" else GaussianH
                Hds = compute_uncertainty(ds,
                                          GaussianH,
                                          order)
                file_name = f"results/{ds_name}_{order}_{seed}.npy"
                np.save(file_name, Hds)



