import numpy as np
import py4DSTEM
import pyprismatic
import pandas as pd
import pathlib

from scipy.ndimage import median_filter, gaussian_filter
from mlr.database.utils import get_new_folder, write_metadata, read_metadata
from ttt.utils import listfiles
from tqdm import tqdm

def get_final_mask_int(mask):
    final_mask = np.abs(mask-1.0)>1e-1 #sim_data are normalized arund mean=1 by definition; finding pixels which deviate
    final_mask = gaussian_filter(final_mask.astype(np.float32), 5.0) > 1e-1 #blur out atom spacing to get smooth mask
    return final_mask

def main():

    base_dir = "/media/files/simulated_databases/defected_Au_np_sims_9_17_2021/simulations/"

    # process base case simulations
    folders = [x for x in listfiles(base_dir) if x.is_dir()]

    # sort folders by creation ID
    folders.sort(key=lambda x: int(str(x).rsplit("_", 2)[-2]))

    rng = np.random.default_rng()

    for f in tqdm(folders):
        
        # load sim data
        sim_data = np.squeeze(py4DSTEM.io.read(f.joinpath("particle_on_substrate.h5"), data_id="HRTEM_virtual").data)

        # load mask data
        mask_data = np.squeeze(py4DSTEM.io.read(f.joinpath("particle.h5"), data_id="HRTEM_virtual").data)

        mask = get_final_mask_int(mask_data).astype(np.uint8)

        metadata = read_metadata(f.joinpath("metadata.json"))

        for dose in [100, 300, 600, 900, 1200]:
            out_folder = get_new_folder(base="dataset/img")

            # apply poisson noise
            noisy_sim_data = pyprismatic.process.apply_poisson_noise_HRTEM(sim_data,
                                                                            (128.0, 128.0),
                                                                            dose=dose,
                                                                            normalize=False).astype(np.uint8)

            # apply median filtering
            noisy_sim_data = median_filter(noisy_sim_data, size=3, mode="wrap")

            # choose a random corner in first 128x128 pixels to start 512x512 crop region
            s = rng.integers(0,128, 2)
            np.save(out_folder.joinpath("train.npy"), noisy_sim_data[s[0]:s[0]+512, s[1]:s[1]+512])
            np.save(out_folder.joinpath("mask.npy"), mask[s[0]:s[0]+512, s[1]:s[1]+512])

            metadata["dose"] = dose
            metadata["median_filter_size"] = 3
            metadata["mask_guassian_filter_size"] = 5.0
            metadata["focal_spread"] = 0
            metadata["C3"] = 0

            write_metadata(metadata, out_folder.joinpath("metadata.json"))

if __name__ == "__main__":
    main()