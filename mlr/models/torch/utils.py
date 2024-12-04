import h5py
import numpy as np
import torch


def is_scalar(arr):
    return np.ndim(arr) == 0


def write_dataset_to_h5(fp, arr, dataset_key, chunks=True, **kwargs):
    with h5py.File(fp, mode="a") as f:
        if chunks is not True:
            max_shape = arr.shape[0]
            chunk_shape = True if max_shape < chunks[0] else chunks
        else:
            chunk_shape = True

        # Write dataset
        if is_scalar(arr):
            dset = f.require_dataset(
                dataset_key, (arr.shape), dtype=arr.dtype, **kwargs
            )
            dset[()] = arr
        else:
            dset = f.require_dataset(
                dataset_key,
                (arr.shape),
                dtype=arr.dtype,
                chunks=chunk_shape,
                compression="gzip",
                **kwargs,
            )
            dset[:] = arr[:]


def write_sd_to_h5(outfile, state_dict):
    for param_name, tensor in state_dict.items():
        temp = tensor.to("cpu")
        write_dataset_to_h5(
            outfile, temp.numpy(), param_name, track_order=True
        )


def read_sd_from_h5(infile):
    out_dict = {}
    with h5py.File(infile, "r") as f:
        for k in f.keys():
            out_dict[k] = torch.from_numpy(np.copy(f[k]))

    return out_dict

def write_eigenpair_to_h5(outfile, evec_dict, eigenvalue):
    write_sd_to_h5(outfile, evec_dict)
    with h5py.File(outfile, mode='a') as f:
        f['/'].attrs['eigenvalue'] = eigenvalue