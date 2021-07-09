import numpy as np


def BernoulliH(recons):
    epsilon = 1e-6
    p = recons.mean(0)
    p = np.clip(p, epsilon, 1 - epsilon)
    h = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return h.sum(0)


def GaussianH(recons):
    std = recons.std(0)
    std = np.clip(std, 1e-1, 10)
    h = (np.log(2 * np.pi * std ** 2) + 1) / 2
    return h.sum(0)


def Hx(ds,
       entropy_func,
       order=None,
       num_samples=100,
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


def compute_uncertainty(ds,
                        entropy_func,
                        order,
                        img_size=64,
                        s=np.random.RandomState()):
    num_factors = ds.num_factors
    images = ds.images.reshape(list(ds.factors_num_values) + list(ds.data_shape))
    z = ds.sample_factors(1, s)[0]
    results = np.zeros((num_factors + 1, img_size, img_size))
    for i in range(num_factors + 1):
        indices = tuple(z[j] if j in order[:i] else slice(None) for j in range(num_factors))
        selected_images = images[indices].reshape([-1]+list(ds.data_shape))
        selected_images = selected_images.transpose(0, 3, 1, 2)
        results[i] = entropy_func(selected_images)
    return results
