from functools import reduce
from itertools import repeat, islice
from time import time

import cupy as cp
import h5py
import numpy as np
import torch
from cupyx.scipy.ndimage import distance_transform_edt
from torch.nn import CrossEntropyLoss, MaxPool2d
from torch.utils.data import DataLoader, Dataset
from ttt.timers import ScopedTimer

from mlr.database.utils import write_metadata
from mlr.models.torch.utils import write_sd_to_h5
from mlr.models.torch.unet import (
    DoubleResidualBlock,
    InitialBlock,
    UNet_v2,
    UpsampleBlock,
)
from mlr.utils.configs import FConfig



#### Dataset utilities
def get_dihedral_augmentations(X):
    if (N_dims := len(X.shape)) == 4:
        axes = (2, 3)
    else:
        axes = (1, 2)
    augmentations = []
    augmentations.append(np.rot90(X, axes=axes, k=1))
    augmentations.append(np.rot90(X, axes=axes, k=2))
    augmentations.append(np.rot90(X, axes=axes, k=3))

    if N_dims == 4:
        flipped = np.concatenate([np.stack([np.fliplr(xx) for xx in x])[None, :, :, :] for x in X])
    else:
        flipped = np.concatenate([np.fliplr(x)[None, :, :] for x in X])

    augmentations.append(flipped)
    augmentations.append(np.rot90(flipped, axes=axes, k=1))
    augmentations.append(np.rot90(flipped, axes=axes, k=2))
    augmentations.append(np.rot90(flipped, axes=axes, k=3))

    return np.concatenate([X] + augmentations)


def normalize_data(X):
    # Center data img wise around 0, std deviation set to 1
    means = np.mean(X, axis=(1, 2))
    X = X - means[:, None, None]
    stds = np.std(X, axis=(1, 2))
    return X / stds[:, None, None]



"""
10 Oct. 2024
(batches are 64 images / 9 sampled images per particle per E0)
This dataset is packed into batches, not necessarily by particle ID,
from a brief inspection, so splitting the training/validation arrays 
by particles will be tough. 

It'll be easier, I think, to just partition over the batch folders for now,
and see if there is a better work around in future.
"""
# def get_particle_id_set(folders):
#     """
#     Packed segmentation database has the data stored as a dense batch,
#     instead of by image. 
    
#     PID set is precalculated and should be partitioned by batches, e.g.,

#     train_set = batch_0 | batch_2
#     vaL_set = batch_1 | batch_3
#     """
#     particle_id_dict = {}
#     for folder in folders:
#         with h5py.File(folder.joinpath("image_batch.h5"), "r") as f:
#             pid_set = set([])
#             for pair_key in [k for k in f.keys() if "img_pair" in k]:
#                 metadata = dict(f[pair_key].attrs)
#                 pid_set.update([metadata["particle_ID"]])

#     return particle_id_dict


def load_dataset(folders, in_set):
    for folder in folders:
        with h5py.File(folder.joinpath("image_batch.h5"), "r") as f:
            if f['img_pair_0'].attrs['E0'] in in_set:
                yield np.array(f['train_batch'])
                yield np.array(f['mask_batch'])


def get_datasets(train_folders, val_folders, E0_set=set([60, 80, 120, 200, 300])):
    res_dict = {}
    res_dict["X_train"] = np.concatenate(
        list(islice(load_dataset(train_folders, E0_set), None, None, 2))
    )
    res_dict["Y_train"] = np.concatenate(
        list(islice(load_dataset(train_folders, E0_set), 1, None, 2))
    )

    res_dict["X_val"] = np.concatenate(
        list(islice(load_dataset(val_folders, E0_set), None, None, 2))
    )
    res_dict["Y_val"] = np.concatenate(
        list(islice(load_dataset(val_folders, E0_set), 1, None, 2))
    )

    return res_dict



def get_level_set(y: np.ndarray, norm: float) -> np.ndarray:
    res = cp.zeros_like(y, dtype=float)

    posmask = cp.array(y).astype(np.bool)
    negmask = ~posmask

    res[0, :, :] = (
        distance_transform_edt(negmask[0, :, :]) * negmask[0, :, :]
        - (distance_transform_edt(posmask[0, :, :]) - 1) * posmask[0, :, :]
    )

    res[1, :, :] = (
        distance_transform_edt(negmask[1, :, :]) * negmask[1, :, :]
        - (distance_transform_edt(posmask[1, :, :]) - 1) * posmask[1, :, :]
    )
    return (res / norm).get()


def prepare_data(dataset):
    subsets = set([s.split("_")[1] for s in dataset.keys()])
    for subset in subsets:
        dataset[f"X_{subset}"] = normalize_data(dataset[f"X_{subset}"].astype(np.float32))

        s = dataset[f"Y_{subset}"].shape
        tmp_y = np.zeros((s[0], 2, *s[1:]), dtype=np.float32)
        tmp_y[:, 0, :, :] = dataset[f"Y_{subset}"]
        tmp_y[:, 1, :, :] = 1.0 - dataset[f"Y_{subset}"]

        dataset[f"Y_{subset}"] = tmp_y

        dataset[f"Y_LS_{subset}"] = np.zeros_like(tmp_y)
        for i in range(tmp_y.shape[0]):
            dataset[f"Y_LS_{subset}"][i, ...] = get_level_set(
                dataset[f"Y_{subset}"][i, ...], norm=256
            )

        dataset[f"Y_LS_offsets_{subset}"] = np.mean(
            dataset[f"Y_{subset}"] * dataset[f"Y_LS_{subset}"], axis=(1, 2, 3)
        )


class PreloadedTupleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return tuple([d[idx] for d in self.data])


class ConvexCombinationLoss:
    """Computes the convex combination for a set of loss functions and corresponding set of weights,

    L(...) = \sum_i w_i * L(...)

    s.t. \sum w_i = 1.0, w_i >= 0 for all i
    """

    def __init__(self, fw_pairs: dict):
        if np.any(weights:= np.array(list(fw_pairs.values()))) < 0:
            raise ValueError(
                "Negative weight value(s) encountered."
            )

        total_weight = np.sum(weights)
        for k in fw_pairs:
            fw_pairs[k] /= total_weight
        self.fw_pairs = fw_pairs

    def __call__(self, *args):
        self.summands = {f:f(*args) for f in self.fw_pairs}
        self.state = {k.__name__:v for k, v in self.summands.items()}

        return reduce(lambda x, y: x + y, (self.fw_pairs[f] * self.summands[f] for f in self.fw_pairs))


cross_entropy_loss = CrossEntropyLoss(reduction="mean")


def cross_entropy(y_hat, y, *args):
    # *args are discared, only y_hat and y are needed
    return cross_entropy_loss(y_hat, y) - 0.3133  # assumes two classes


def boundary_loss(y_hat, y, y_ls, y_ls_offset, *args):
    return torch.mean(torch.mean(y_hat * y_ls, axis=(1, 2, 3)) - y_ls_offset)


def ising_regularization(y_hat, *args):
    res = 2 * (torch.sigmoid(10 * (y_hat - 0.5)) - 0.5)  # set outputs to be +1 or -1
    H = (
        -1
        * (
            torch.sum(res[:, :, :-1, :] * res[:, :, 1:, :])
            + torch.sum(res[:, :, :, :-1] * res[:, :, :, 1:])
        )
        / torch.numel(res)
    )

    return torch.exp(H / 0.5)  # 0.5 can be adjust to affect the 'temperature"


def get_model_config():
    N_blocks = 4
    dfilters = [64, 128, 256, 512]
    ufilters = [256, 128, 64, 32]
    first_filter = 64
    final_filter = 16
    N_classes = 2

    # Input filter dimension for U_i = size(D_[N_d - i]) + size(U_[i-1])
    internal = [u + d for u, d in zip(ufilters[:-2], dfilters[:-2][::-1])]
    concat_sizes = [dfilters[-2] + dfilters[-1]] + internal + [first_filter + ufilters[-2]]

    initial_config = FConfig(InitialBlock, stride=int, bias=bool)(first_filter, 7, 2, False)
    initial_pool_config = FConfig(MaxPool2d, stride=int)(2, 2)
    encoder_fconfig = FConfig(DoubleResidualBlock, bias=bool)
    decoder_fconfig = FConfig(UpsampleBlock, bias=bool)

    encoder_configs = [
        encoder_fconfig(*args)
        for args in zip(
            [first_filter] + dfilters[:-1],
            dfilters,
            repeat(3, N_blocks),
            repeat(3, N_blocks),
            [1, 2, 2, 2],
            repeat(False, N_blocks),
        )
    ]

    decoder_configs = [
        decoder_fconfig(*args)
        for args in zip(
            concat_sizes,
            ufilters,
            repeat(3, N_blocks),
            repeat(3, N_blocks),
            repeat(2, N_blocks),
            repeat(False, N_blocks),
        )
    ]

    final_config = decoder_fconfig(
        decoder_configs[-1]["out_channels"], final_filter, 3, 3, 2, False
    )
    final_conv_config = FConfig(torch.nn.Conv2d, padding=int)(final_filter, N_classes, 3, 1)

    return {
        "num_blocks": N_blocks,
        "initial_config": initial_config,
        "initial_pool_config": initial_pool_config,
        "encoder_configs": encoder_configs,
        "decoder_configs": decoder_configs,
        "final_config": final_config,
        "final_conv_config": final_conv_config,
    }


def get_model():
    return UNet_v2(**get_model_config())


def training_loop(
    output_folder,
    train_dataset,
    val_dataset,
    model,
    optimizer,
    criterion,
    scheduler,
    N_epochs,
    verbose=True,
    save_every_epoch=True,
    **kwargs,
):
    train_dataloader = DataLoader(train_dataset, **kwargs)
    val_dataloader = DataLoader(val_dataset, **kwargs)

    N_train_batches = len(train_dataloader)
    N_val_batches = len(val_dataloader)

    history_dict = {}
    for epoch in range(N_epochs):
        t0 = time()
        with ScopedTimer("FULL EPOCH"):
            running_loss = {"training": {}, "validation": {}}

            # Update model on training data
            model.train()
            for train_data in train_dataloader:
                # Reset state
                optimizer.zero_grad()

                # Forward pass
                t_X = train_data[0]
                t_Y_hat = model.forward(t_X)

                # Measure loss and calculate updates
                train_loss = criterion(t_Y_hat, *train_data[1:])
                train_loss.backward()
                optimizer.step()

                # Store metrics
                if len(running_loss["training"]) == 0:
                    running_loss["training"] = {
                        name: loss.item() for name, loss in criterion.state.items()
                    }
                    running_loss["training"]["total"] = train_loss.item()
                    loss_dict = running_loss["training"]
                else:
                    for name, loss in criterion.state.items():
                        loss_dict[name] += loss.item()
                    loss_dict["total"] += train_loss.item()

            # Post update callbacks
            model.eval()
            loss_dict = running_loss["validation"]
            with torch.no_grad():
                for val_data in val_dataloader:
                    # Forward pass
                    v_X = val_data[0]
                    v_Y_hat = model.forward(v_X)

                    # Measure loss and store metric
                    val_loss = criterion(v_Y_hat, *val_data[1:])
                    if len(running_loss["validation"]) == 0:
                        running_loss["validation"] = {
                            name: loss.item() for name, loss in criterion.state.items()
                        }
                        running_loss["validation"]["total"] = val_loss.item()
                        loss_dict = running_loss["validation"]
                    else:
                        for name, loss in criterion.state.items():
                            loss_dict[name] += loss.item()
                        loss_dict["total"] += val_loss.item()

            for k, v in running_loss["training"].items():
                running_loss["training"][k] = v / N_train_batches

            for k, v in running_loss["validation"].items():
                running_loss["validation"][k] = v / N_val_batches

            scheduler.step()

        # Print end of epoch summary
        t1 = time()
        if verbose:
            summary_string = f"Epoch {epoch + 1:04}/{N_epochs:04} summary"
            training_string = f" total training loss : {running_loss['training']['total']:6.4f} total validation loss : {running_loss['validation']['total']:6.4f}"
            title_string = f"{summary_string:<25}{training_string:>60}"
            print(title_string)
            component_keys = set(running_loss["training"].keys()).difference(["total"])
            for k in sorted(list(component_keys)):
                t_perf = running_loss["training"][k]
                v_perf = running_loss["validation"][k]

                component_summary = (
                    f" ------- {k} | training : {t_perf:6.4f} validation : {v_perf:6.4f}"
                )
                print(f"{component_summary:>85}")

        if save_every_epoch:
            write_metadata(history_dict, output_folder.joinpath("history.json"))
            write_sd_to_h5(output_folder.joinpath(f'epoch_{epoch}_weights.h5'), model.state_dict())

        history_dict[epoch] = {"time": t1 - t0, **running_loss}

    return history_dict
