"""
Code for resampling of images.
"""

import numpy as np
import cupy as cp
from pyfftw.interfaces.numpy_fft \
     import fft2 as fftw_fft2, ifft2 as fftw_ifft2
from cupy.fft import fft2 as cupy_fft2, ifft2 as cupy_ifft2

def fourier_downsample(img, target_shape, backend="fftw"):
    """Downsample an image using Fourier resampling.

    Args:
        img (np.ndarray): 2D numpy array to downsample
        target_shape List[int, int]: target shape to downsample to. 
                                    Must be smaller than img.shape
        backend (str): which FFT backend to use. 
                        options are "numpy", "fftw", or "cupy"

    Returns:
        Downsampled image as numpy array, in original type if not double precision.

    """
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

def fourier_downsample_batch(img, target_shape, backend="fftw", axes=None):
    """ Downsample a stack of images using Fourier resampling.

    Args:
        img (np.ndarray): 3D numpy array to downsample along two dimensions.
        target_shape List[int, int]: target shape to downsample to. 
                                    Must be smaller than img.shape for relevant
                                    axes.
        backend (str): which FFT backend to use. 
                        options are "numpy", "fftw", or "cupy", defaults "fftw".
        axes List[int, int]: optional, decide which axes FFT is performed over.
                            defaults to (0,1)

    Returns:
        Downsampled stack as numpy array, in original type if not double precision.
    
    """

    os = img.shape
    ts = target_shape
    dtype = img.dtype

    if axes is None:
        axes = (0,1)

    if backend=="cupy":
        img = cp.asarray(img)

    # fourier transform image
    if backend=="cupy":
        k_img = cupy_fft2(img, axes=axes)
    elif backend=="fftw":
        k_img = fftw_fft2(img, axes=axes)
    elif backend=="numpy":
        k_img = np.fft.fft2(img, axes=axes)

    # fft-shift and roll 
    # k_img = np.fft.fftshift(k_img)
    k_img = np.roll(k_img, axis=axes, shift=(ts[0]//2, ts[1]//2))

    # get set difference of axes to create proper slicing for arbitrary data ordering

    ind_axes = (*axes, tuple({0,1,2}.difference(axes))[0])
    ind = [0, 0, 0]
    ind[ind_axes[0]] = slice(0, ts[0])
    ind[ind_axes[1]] = slice(0, ts[1])
    ind[ind_axes[2]] = slice(0, None)
    ind = tuple(ind)

    # fourier transform cropped region
    if backend=="cupy":
        r_img = cupy_ifft2(k_img[ind], axes=axes)
    elif backend=="fftw":
        r_img = fftw_ifft2(k_img[ind], axes=axes)
    elif backend=="numpy":
        r_img = np.fft.ifft2(k_img[ind], axes=axes)

    # renormalize
    r_img *= (ts[0]/os[axes[0]])*(ts[1]/os[axes[1]])
    r_img = np.abs(r_img).astype(dtype)
    
    if backend=="cupy":
        r_img = cp.asnumpy(r_img)

    return r_img