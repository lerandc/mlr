import sys
from contextlib import redirect_stdout
from itertools import repeat
from pathlib import Path
from time import time

import h5py
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MaxPool2d
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from mlr.database.utils import get_new_folder_parallel, write_metadata
from mlr.models.torch.unet import (DoubleResidualBlock, InitialBlock, UNet_v2,
                                   UpsampleBlock)
from mlr.models.torch.utils import read_sd_from_h5, write_sd_to_h5
from mlr.utils.configs import FConfig
from ttt.timers import ScopedTimer
from ttt.utils import listfolders


#### Dataset utilities
class PreloadedSegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def normalize_data(X):
    # Center data img wise around 0, std deviation set to 1
    means = np.mean(X, axis=(1, 2))
    X = X - means[:, None, None]
    stds = np.std(X, axis=(1, 2))
    return X / stds[:, None, None]


def get_particle_id_set(folders):
    particle_ids = {}
    for folder in folders:
        with h5py.File(folder.joinpath("image_batch.h5"), "r") as f:
            for k in f.keys():
                metadata = dict(f[k].attrs)
                pid = metadata["particle_ID"]
                defocus = metadata["target_C1"]

                if pid not in particle_ids:
                    particle_ids[pid] = set([defocus])
                else:
                    particle_ids[pid].update([defocus])

    return particle_ids


def load_dataset(folders, group_set):
    for folder in folders:
        with h5py.File(folder.joinpath("image_batch.h5"), "r") as f:
            for k in f.keys():
                in_group = f[k].attrs["particle_ID"] in group_set
                defocus = f[k].attrs["target_C1"]
                yield defocus, in_group, np.array(f[k]["X"]), np.array(f[k]["Y"])


def conditional_load_dataset(folders, group_set, target_key):
    for folder in folders:
        with h5py.File(folder.joinpath("image_batch.h5"), "r") as f:
            for k in f.keys():
                in_group = f[k].attrs["particle_ID"] in group_set
                defocus = f[k].attrs["target_C1"]
                if (defocus in target_key) or in_group:
                    yield defocus, in_group, np.array(f[k]["X"]), np.array(f[k]["Y"])


def partial_load_dataset(folders, training_group, validation_group):
    for folder in folders:
        with h5py.File(folder.joinpath("image_batch.h5"), "r") as f:
            for k in f.keys():
                in_train = f[k].attrs["particle_ID"] in training_group
                in_val = f[k].attrs["particle_ID"] in validation_group
                defocus = f[k].attrs["target_C1"]
                if in_train or in_val:
                    yield defocus, in_train, np.array(f[k]["X"]), np.array(f[k]["Y"])

def class_partial_load_dataset(folders, training_group_maps, validation_group):
    for folder in folders:
        with h5py.File(folder.joinpath("image_batch.h5"), "r") as f:
            for k in f.keys():
                defocus = f[k].attrs["target_C1"]
                in_train = f[k].attrs["particle_ID"] in training_group_maps[defocus]
                in_val = f[k].attrs["particle_ID"] in validation_group
                if in_train or in_val:
                    yield defocus, in_train, np.array(f[k]["X"]), np.array(f[k]["Y"])


def get_datasets(batch_folders, train_ids, val_ids):
    N_training = len(train_ids)
    N_validation = len(val_ids)

    def initialize_train_arrays():
        return np.zeros((N_training, 512, 512), dtype=np.uint8), np.zeros(
            (N_training, 512, 512), dtype=np.uint8
        )

    def initialize_validation_arrays():
        return np.zeros((N_validation, 512, 512), dtype=np.uint8), np.zeros(
            (N_validation, 512, 512), dtype=np.uint8
        )

    res_dict = {}
    for defocus, in_training_set, X, Y in load_dataset(batch_folders, train_ids):
        if defocus not in res_dict:
            res_dict[defocus] = (dataset := {})
            dataset["X_train"], dataset["Y_train"] = initialize_train_arrays()
            dataset["X_validation"], dataset["Y_validation"] = (
                initialize_validation_arrays()
            )
            dataset["N_train"], dataset["N_validation"] = 0, 0
        else:
            dataset = res_dict[defocus]

        key = "_train" if in_training_set else "_validation"
        idx = dataset[f"N{key}"]
        dataset[f"X{key}"][idx, :, :] = X
        dataset[f"Y{key}"][idx, :, :] = Y
        dataset[f"N{key}"] += 1

    return res_dict


def conditional_get_datasets(batch_folders, train_ids, val_ids, target_key):

    N_training = len(train_ids)
    N_validation = len(val_ids)

    def initialize_train_arrays():
        return np.zeros((N_training, 512, 512), dtype=np.uint8), np.zeros(
            (N_training, 512, 512), dtype=np.uint8
        )

    def initialize_validation_arrays():
        return np.zeros((N_validation, 512, 512), dtype=np.uint8), np.zeros(
            (N_validation, 512, 512), dtype=np.uint8
        )

    res_dict = {}
    for defocus, in_validation_set, X, Y in conditional_load_dataset(
        batch_folders, val_ids, target_key
    ):
        if defocus not in res_dict:
            res_dict[defocus] = (dataset := {})
            if defocus in target_key:
                dataset["X_train"], dataset["Y_train"] = initialize_train_arrays()
                dataset["N_train"] = 0
            dataset["X_validation"], dataset["Y_validation"] = (
                initialize_validation_arrays()
            )
            dataset["N_validation"] = 0
        else:
            dataset = res_dict[defocus]

        key = "_validation" if in_validation_set else "_train"
        idx = dataset[f"N{key}"]
        dataset[f"X{key}"][idx, :, :] = X
        dataset[f"Y{key}"][idx, :, :] = Y
        dataset[f"N{key}"] += 1

    return res_dict


def partial_get_datasets(batch_folders, train_ids, val_ids):
    N_training = len(train_ids)
    N_validation = len(val_ids)

    def initialize_arrays(N):
        return np.zeros((N, 512, 512), dtype=np.uint8), np.zeros(
            (N, 512, 512), dtype=np.uint8
        )

    res_dict = {}
    for defocus, in_train_set, X, Y in partial_load_dataset(
        batch_folders, train_ids, val_ids
    ):
        if defocus not in res_dict:
            res_dict[int(defocus)] = (dataset := {})
            dataset["X_train"], dataset["Y_train"] = initialize_arrays(N_training)
            dataset["N_train"] = 0
            dataset["X_validation"], dataset["Y_validation"] = initialize_arrays(
                N_validation
            )
            dataset["N_validation"] = 0
        else:
            dataset = res_dict[defocus]

        key = "_train" if in_train_set else "_validation"
        idx = dataset[f"N{key}"]
        dataset[f"X{key}"][idx, :, :] = X
        dataset[f"Y{key}"][idx, :, :] = Y
        dataset[f"N{key}"] += 1

    return res_dict

def class_partial_get_datasets(batch_folders, train_ids_map, val_ids):
    N_validation = len(val_ids)

    def initialize_arrays(N):
        return np.zeros((N, 512, 512), dtype=np.uint8), np.zeros(
            (N, 512, 512), dtype=np.uint8
        )

    res_dict = {}
    for defocus, in_train_set, X, Y in class_partial_load_dataset(
        batch_folders, train_ids_map, val_ids
    ):
        if defocus not in res_dict:
            res_dict[int(defocus)] = (dataset := {})
            N_training = len(train_ids_map[defocus])
            dataset["X_train"], dataset["Y_train"] = initialize_arrays(N_training)
            dataset["N_train"] = 0
            dataset["X_validation"], dataset["Y_validation"] = initialize_arrays(
                N_validation
            )
            dataset["N_validation"] = 0
        else:
            dataset = res_dict[defocus]

        key = "_train" if in_train_set else "_validation"
        idx = dataset[f"N{key}"]
        dataset[f"X{key}"][idx, :, :] = X
        dataset[f"Y{key}"][idx, :, :] = Y
        dataset[f"N{key}"] += 1

    return res_dict



def prepare_data(dataset_dict):
    for dataset in dataset_dict.values():
        subsets = set([s.split("_")[1] for s in dataset.keys()])
        for subset in subsets:
            dataset[f"X_{subset}"] = normalize_data(
                dataset[f"X_{subset}"].astype(np.float32)
            )

            s = dataset[f"Y_{subset}"].shape
            tmp_y = np.zeros((s[0], 2, *s[1:]), dtype=np.float32)
            tmp_y[:, 0, :, :] = dataset[f"Y_{subset}"]
            tmp_y[:, 1, :, :] = 1.0 - dataset[f"Y_{subset}"]

            dataset[f"Y_{subset}"] = tmp_y


def prepare_callbacks(datasets, device):
    for v in datasets.values():
        v["callback_dataset"] = PreloadedSegmentationDataset(
            torch.from_numpy(v["X_validation"][:, None, :, :]).to(device),
            torch.from_numpy(v["Y_validation"]).to(device),
        )


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
        flipped = np.concatenate(
            [np.stack([np.fliplr(xx) for xx in x])[None, :, :, :] for x in X]
        )
    else:
        flipped = np.concatenate([np.fliplr(x)[None, :, :] for x in X])

    augmentations.append(flipped)
    augmentations.append(np.rot90(flipped, axes=axes, k=1))
    augmentations.append(np.rot90(flipped, axes=axes, k=2))
    augmentations.append(np.rot90(flipped, axes=axes, k=3))

    return np.concatenate([X] + augmentations)


#### Model utilities
def get_model_config():
    N_blocks = 4
    dfilters = [64, 128, 256, 512]
    ufilters = [256, 128, 64, 32]
    first_filter = 64
    final_filter = 16
    N_classes = 2

    # Input filter dimension for U_i = size(D_[N_d - i]) + size(U_[i-1])
    internal = [u + d for u, d in zip(ufilters[:-2], dfilters[:-2][::-1])]
    concat_sizes = (
        [dfilters[-2] + dfilters[-1]] + internal + [first_filter + ufilters[-2]]
    )

    initial_config = FConfig(InitialBlock, stride=int, bias=bool)(
        first_filter, 7, 2, False
    )
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
    final_conv_config = FConfig(torch.nn.Conv2d, padding=int)(
        final_filter, N_classes, 3, 1
    )

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


#### Training utilities
def training_loop(
    dataset_dict,
    model,
    optimizer,
    scheduler,
    criterion,
    N_epochs,
    verbose=True,
    dump_folder=None,
    save_weights=False,
    offset=0,
    **kwargs,
):

    print(kwargs)
    train_dataset = dataset_dict["target"]["dataset"]
    train_dataloader = DataLoader(train_dataset, **kwargs)

    N_train_batches = len(train_dataloader)

    val_dataloaders = {}
    for defocus, dataset in dataset_dict.items():
        if defocus == "target":
            continue
        val_dataloaders[defocus] = DataLoader(dataset["callback_dataset"], **kwargs)

    target_key = f"training_{dataset_dict['target']['defocus']}"
    history_dict = {}

    for epoch in range(N_epochs):
        t0 = time()
        with ScopedTimer("FULL EPOCH"):
            running_loss = {target_key: 0.0}

            with ScopedTimer("training"):
                # Update model on training data
                model.train()
                for train_data in train_dataloader:
                    # Reset state
                    optimizer.zero_grad()

                    # Forward pass
                    t_X, t_Y = train_data
                    t_Y_hat = model.forward(t_X)

                    # Measure loss and calculate updates
                    train_loss = criterion(t_Y_hat, t_Y)
                    train_loss.backward()

                    optimizer.step()

                    # Store metrics
                    running_loss[target_key] += train_loss.item()

                running_loss[target_key] /= N_train_batches

            with ScopedTimer("callbacks"):
                # Post update callbacks
                model.eval()
                with torch.no_grad():
                    for defocus, cur_dataloader in val_dataloaders.items():
                        running_loss[f"validation_{defocus}"] = 0.0
                        N_val_batches = len(cur_dataloader)
                        for val_data in cur_dataloader:
                            # Forward pass
                            v_X, v_Y = val_data
                            v_Y_hat = model.forward(v_X)

                            # Measure loss and store metric
                            val_loss = criterion(v_Y_hat, v_Y)
                            running_loss[f"validation_{defocus}"] += val_loss.item()

                        running_loss[f"validation_{defocus}"] /= N_val_batches

            scheduler.step()

        # Print end of epoch summary
        t1 = time()
        if verbose:
            summary_string = f"Epoch {epoch + 1:04}/{N_epochs:04} summary"
            training_string = f"{target_key} : {running_loss[target_key]-offset:6.4f}"
            title_string = f"{summary_string:<35}{training_string:>25}"
            print(title_string)
            val_keys = set(running_loss.keys()).difference([target_key])
            val_mean = np.mean([running_loss[k] for k in val_keys])         
            val_summary = f" ---------- {'val_mean':>20} : {val_mean - offset:6.4f}"
            print(f"{val_summary:>60}")
            for k in sorted(val_keys, key=lambda x: int(x.split("_")[-1])):
                performance = running_loss[k]
                if k != target_key:
                    val_summary = f" ---------- {k:>20} : {performance-offset:6.4f}"
                    print(f"{val_summary:>60}")


        history_dict[epoch] = {"time": t1 - t0, **running_loss}
        if dump_folder is not None:
            write_metadata(history_dict, dump_folder.joinpath("history_dump.json"))

            if save_weights:
                write_sd_to_h5(dump_folder.joinpath(f"weights_epoch_{epoch}.h5"), model.state_dict())
        

    return history_dict

def run_training(
    output_folder,
    device,
    target_key,
    datasets,
    train_data,
    lr=1e-1,
    N_training_points=-1,
    schedule_rate=0.98,
    batch_size=16,
    **kwargs,
):
    torch._dynamo.reset()
    with open(output_folder.joinpath("log.out"), "w", buffering=1) as f:
        with redirect_stdout(f):
            print("Starting training.")
            with ScopedTimer("Model preparation."):
                print("Initializing model.")
                model = get_model()

                print("Compiling model.")
                opt_model = torch.compile(
                    model, backend="inductor", mode="max-autotune"
                )

                # print("Initalizing compiled model.")
                opt_model.to(device)
                tmp_X = torch.from_numpy(
                    datasets[target_key]["X_train"][:batch_size, None, :, :]
                ).to(device)
                for _ in range(5):
                    _ = opt_model(tmp_X)
                del tmp_X

                print("Setting up optimizer.")
                optimizer = SGD(opt_model.parameters(), lr=lr)
                scheduler = StepLR(optimizer, step_size=1, gamma=schedule_rate)
                criterion = CrossEntropyLoss()

            training_dict = {
                **datasets,
                "target": {"defocus": target_key, "dataset": train_data},
            }

            print("Beginning training loop.")
            training_history = training_loop(
                training_dict,
                opt_model,
                optimizer,
                scheduler,
                criterion,
                batch_size=batch_size,
                dump_folder=output_folder,
                **kwargs,
            )

    metadata = {
        "lr": lr,
        "batch_size": batch_size,
        "target_key": target_key,
        **kwargs,
        "history": {**training_history},
    }
    write_metadata(metadata, output_folder.joinpath("metadata.json"))
