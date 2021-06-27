import numpy as np

def BernoulliH(recons):
    epsilon = 1e-6
    p = recons.mean(0)
    p = np.clip(p,epsilon,1-epsilon)
    h = - (p * np.log(p) + (1-p) * np.log(1-p) )
    return h.sum(0)

def GaussianH(recons):
    std = recons.std(0)
    std = np.clip(std,1e-1,10)
    h = (np.log(2*np.pi*std**2)+1)/2
    return h.sum(0)

def Hx(ds,
       entropy_func,
       order = None,
       num_samples=100,
       img_size=64):
    s = np.random.RandomState()
    num_latent = ds.num_factors
    results = np.zeros((num_latent + 1, img_size, img_size))
    z = ds.sample_factors(1,s)
    if order is None:
        order = list(range(num_latent))
    for i in range(0, num_latent + 1):
        r = ds.sample_factors(num_samples,s)
        zz = np.repeat(z.reshape(1,-1),num_samples,axis=0)
        r[:,order[:i]] = zz[:,order[:i]]
        recons = ds.sample_observations_from_factors(r,s).transpose(0,3,1,2)
        results[i] = entropy_func(recons)
    return results