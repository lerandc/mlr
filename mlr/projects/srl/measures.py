import numpy as np

def s2(X: np.ndarray, K: np.ndarray):
    """
    Self-normalized estimator for 2-body structure factor S^2.

    Args:
        X: set of point samples
        K: wave vectors K over which to calculate 
    
    """
    sigma = K @ X.T
    rho_k = np.sum(np.exp(-1j*sigma), axis=1)
    s_k = np.abs(rho_k)**2.0

    return s_k / (X.shape[0])


def s3(X: np.ndarray, K1: np.ndarray, K2: np.ndarray):
    """
    Self-normalized estimator for 3-body structure factor S^3(K, K').
    
    Args:
        X: set of point samples
        K1: set of wave vectors
        K2: set of wave vectors
    """
    res = np.zeros((K1.shape[0], K2.shape[0]), dtype=np.complex128)

    res_0 = np.sum(np.exp(-1j * (K1 @ X.T)), axis=-1)
    res_1 = np.sum(np.exp(-1j * (K2 @ X.T)), axis=-1)
    res_2 = np.sum(np.exp(-1j * -(K1[:,None,:] + K2[None,:,:]) @ X.T), axis=-1)

    res[:,:] = res_0[:,None] * res_1[None, :] * res_2

    return res/X.shape[0]
