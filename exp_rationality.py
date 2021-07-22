# %env DISENTANGLEMENT_LIB_DATA=/home/ubuntu/work/wjt/disentanglement_lib

from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
from collections import defaultdict
from uncertainty_estimation import *
results = defaultdict(lambda: defaultdict(dict))


uncertainty_gap = []
order = [3,1,2,0]
ds = get_named_ground_truth_data('smallnorb')

for j in range( ds.num_factors+1):
    mix_factors = order[:j]
    entropies = []
    for i in range(10):
        s = np.random.RandomState(i)
        if j >0:
            entropy = compute_uncertainty_mix_variable(ds, entropy_gaussian, mix_factors,s=s)
        else:
            r= ds.sample_factors(10000,s)
            recons = ds.sample_observations_from_factors(r, s).transpose(0, 3, 1, 2)
            entropy = entropy_gaussian(recons)
        results[ds.__class__.__name__][j][i] = entropy
        entropies.append(entropy)
    uncertainty_gap.append(entropies)
uncertainty_gap = np.array(uncertainty_gap)

