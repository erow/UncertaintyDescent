import numpy as np
from disentanglement_lib.data.ground_truth.ground_truth_data import GroundTruthData

epsilon = 1e-5


def entropy_bernoulli(recons: np.array):
    """

    :type recons: np.array, sample x channel ...
    """
    p = recons.mean(0)
    p = np.clip(p, epsilon, 1 - epsilon)
    h = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return h.mean(0)


def entropy_gaussian(recons: np.array):
    """

    :type recons: np.array, sample x channel ...
    """
    std = recons.std(0)
    std = np.clip(std, epsilon, 10)
    h = (np.log(2 * np.pi * std * std) + 1) / 2
    return h.mean(0)


def compute_uncertainty(ds,
                        entropy_func,
                        order=None,
                        num_samples=10000,
                        img_size=64,
                        s=np.random.RandomState()):
    num_factors = ds.num_factors
    results = np.zeros((num_factors + 1, img_size, img_size))
    z = ds.sample_factors(1, s)
    if order is None:
        order = list(range(num_factors))
    for i in range(0, num_factors + 1):
        r = ds.sample_factors(num_samples, s)
        zz = np.repeat(z.reshape(1, -1), num_samples, axis=0)
        r[:, order[:i]] = zz[:, order[:i]]
        recons = ds.sample_observations_from_factors(r, s).transpose(0, 3, 1, 2)
        results[i] = entropy_func(recons)
    return results


# def compute_uncertainty(ds: GroundTruthData,
#                         entropy_func,
#                         order,
#                         img_size=64,
#                         s=np.random.RandomState()):
#     num_factors = ds.num_factors
#     images = ds.images.reshape(list(ds.factors_num_values) + list(ds.observation_shape))
#     z = ds.sample_factors(1, s)[0]
#     results = np.zeros((num_factors + 1, img_size, img_size))
#     for i in range(num_factors + 1):
#         indices = tuple(z[j] if j in order[:i] else slice(None) for j in range(num_factors))
#         selected_images = images[indices].reshape([-1]+list(ds.observation_shape))
#         selected_images = selected_images.transpose(0, 3, 1, 2)
#         results[i] = entropy_func(selected_images)
#     return results


def dependency_count(order):
    count = np.zeros((5, 5))
    for row in order:
        for i in range(5):
            for j in range(i + 1, 5):
                count[row[i], row[j]] += 1
    return count


def perm_codebook(factor_sizes):
    factor_sizes = list(factor_sizes)
    indices = np.random.permutation(np.prod(factor_sizes))
    factor_bases = np.prod(factor_sizes) / np.cumprod(
        factor_sizes)
    codebook = []
    for i in range(len(factor_sizes)):
        codebook.append((indices // factor_bases[i]) % factor_sizes[i])
    return np.stack(codebook, 1).reshape(factor_sizes + [-1])


def compute_uncertainty_mix_variable(ds: GroundTruthData,
                                     entropy_func,
                                     mix_factors: [int],
                                     img_size=64,
                                     num_samples=10000,
                                     s=np.random.RandomState()):
    """mix_factors 传入要混合的因子。
    默认计算mix第一个变量的uncertainty gap."""
    num_factors = ds.num_factors
    factors_num_values = np.array(ds.factors_num_values)
    z = ds.sample_factors(1, s)[0]
    permutation = perm_codebook(factors_num_values[mix_factors])
    r = ds.sample_factors(num_samples, s)
    # incorporate new variable
    known_variables = permutation[z[mix_factors[0]]]
    known_variables = known_variables.reshape(-1, len(mix_factors))

    # sample the unknown part
    sampled_variables = known_variables[np.random.randint(len(known_variables), size=(num_samples,))]

    # replace with permuted codebook
    r[:, mix_factors] = sampled_variables
    recons = ds.sample_observations_from_factors(r, s).transpose(0, 3, 1, 2)
    return entropy_func(recons)
