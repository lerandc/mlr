import numpy as np

def sample_normal(rng, mean, std, shape=(1)):
    """Sample normal distribution with numpy rng.

    Args:
        rng : random number generator class from numpy.random
        shape Tuple[int,]: desired number of samples, defaults to (1)
        mean (float): mean of distribution
        std (float): std deviation of distribution
    
    """
    samples = mean+std*rng.standard_normal(shape)
    return samples