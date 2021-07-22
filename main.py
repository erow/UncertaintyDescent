from __future__ import print_function, division

from itertools import permutations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from disentanglement_lib.data.ground_truth import named_data
from uncertainty_estimation import *

if __name__ == '__main__':
    for ds_name in ['dsprites_full']:
        ds = named_data.get_named_ground_truth_data(ds_name)
        for seed in range(3):
            for i,order in enumerate(permutations(range(ds.num_factors))):
                entropy_func =  entropy_bernoulli if ds_name == "dsprites_full" else entropy_gaussian
                Hds = compute_uncertainty(ds,
                                          entropy_gaussian,
                                          order)
                file_name = f"results/{ds_name}_{order}_{seed}.npy"
                print(file_name)
                np.save(file_name, Hds)



