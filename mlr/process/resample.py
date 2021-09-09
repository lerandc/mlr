"""
Code for resampling of images.
"""

import numpy as np
import cupy as cp
from pyfftw.interfaces.numpy_fft \
     import fft2 as fftw_fft2, ifft2 as fftw_ifft2
from cupy.fft import fft2 as cupy_fft2, ifft2 as cupy_ifft2

def fourier_resample(img, target_shape, backend="fftw"):
    os = img.shape
    ts = target_shape
    dtype = img.dtype

    if backend=="cupy":
        img = cp.asarray(img)

    # fourier transform image
    if backend=="cupy":
        k_img = cupy_fft2(img)
    elif backend=="fftw":
        k_img = fftw_fft2(img)
    elif backend=="numpy":
        k_img = np.fft.fft2(img)

    # fft-shift and roll 
    # k_img = np.fft.fftshift(k_img)
    k_img = np.roll(k_img, axis=(0,1), shift=(ts[0]//2, ts[1]//2))

    # fourier transform cropped region
    if backend=="cupy":
        r_img = cupy_ifft2(k_img[:ts[0], :ts[1]])
    elif backend=="fftw":
        r_img = fftw_ifft2(k_img[:ts[0], :ts[1]])
    elif backend=="numpy":
        r_img = np.fft.ifft2(k_img[:ts[0], :ts[1]])

    # renormalize
    r_img *= (ts[0]/os[0])*(ts[1]/os[1])
    r_img = np.abs(r_img).astype(dtype)
    
    if backend=="cupy":
        r_img = cp.asnumpy(r_img)

    return r_img