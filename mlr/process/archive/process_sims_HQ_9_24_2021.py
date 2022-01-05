import numpy as np
import py4DSTEM
import pyprismatic
import cupy as cp

from scipy.ndimage import median_filter, gaussian_filter
from mlr.database.utils import get_new_folder, write_metadata, read_metadata
from mlr.sampling.distribution import sample_normal
from ttt.utils import listfiles
from tqdm import tqdm
from cupy.fft import fft2 as cupy_fft2, ifft2 as cupy_ifft2
from itertools import product

def get_final_mask_int(mask):
    final_mask = np.abs(mask-1.0)>1e-1 #sim_data are normalized arund mean=1 by definition; finding pixels which deviate
    final_mask = gaussian_filter(final_mask.astype(np.float32), 5.0) > 1e-1 #blur out atom spacing to get smooth mask
    return final_mask

def fast_apply_aberration(kpsi, chi):
    """
    input kpsi in fourier_space, on device
    chi in fourier space, on device

    output a_psi on host
    """
    # kpsi = np.fft.fft2(psi) #bring wave fucntion to fourier space
    a_kpsi = kpsi*cp.exp(-1j*chi) #apply aberration in fourier space
    a_psi = cupy_ifft2(a_kpsi) #transorm aberrated psi back to real space
    return cp.asnumpy(a_psi)


def main():

    base_dir = "/media/files/simulated_databases/defected_Au_np_sims_HQ_9_17_2021/simulations/"

    # process base case simulations
    folders = [x for x in listfiles(base_dir) if x.is_dir()]

    # sort folders by creation ID
    folders.sort(key=lambda x: int(str(x).rsplit("_", 2)[-2]))

    rng = np.random.default_rng()

    defocus_sigma = 15
    defoci = np.linspace(-(3*defocus_sigma), 3*defocus_sigma+1, 25)
    defoci_weights = pyprismatic.process.rad_weights(defoci, defocus_sigma, np.ones(defoci.shape))

    for f in tqdm(folders):
        
        metadata = read_metadata(f.joinpath("metadata.json"))
        # load all sim data
        sim_data = np.zeros((metadata["numFP"], 640, 640), dtype=np.complex64)
        for i in range(metadata["numFP"]):
            sim_data[i,:,:] = np.squeeze(py4DSTEM.io.read(f.joinpath("particle_on_substrate.h5"), data_id="HRTEM_virtual_fp0000").data)

        # load mask data
        mask_data = np.zeros((metadata["numFP"], 640, 640), dtype=np.complex64)
        for i in range(metadata["numFP"]):
            mask_data[i,:,:] = np.squeeze(py4DSTEM.io.read(f.joinpath("particle.h5"), data_id="HRTEM_virtual_fp0000").data)

        # load data onto device and fourier transform
        sim_data_d = cp.asarray(sim_data)
        mask_data_d = cp.asarray(mask_data)

        ksim_data_d = cupy_fft2(sim_data_d, axes=(1,2))
        kmask_data_d = cupy_fft2(mask_data_d, axes=(1,2))

        # get relevant arrays for applying aberrations
        q_lambda = pyprismatic.process.get_lambda(f.joinpath("particle_on_substrate.h5"))
        q_arr = py4DSTEM.io.read(f.joinpath("particle_on_substrate.h5"), data_id="qArr").data

        q = np.sqrt(q_arr[:,:,0]**2.0 + q_arr[:,:,1]**2.0)
        qTheta = np.arctan2(q_arr[:,:,1], q_arr[:,:,0])

        # set loop variables
        dose_list = [100, 300, 600, 900, 1200]
        df_list = sample_normal(rng, 30, 20, 4)

        for dose, defocus in tqdm(list(product(*(dose_list, df_list))), leave=False):

            # apply defocus to mask FPs; incoherently average and form final mask
            chi_d = cp.asarray(pyprismatic.process.calc_defocus_aberration(q, qTheta, q_lambda, defocus).astype(np.complex64)[None,:,:])
            df_mask = fast_apply_aberration(kmask_data_d, chi_d)
            mask = get_final_mask_int(np.mean(np.abs(df_mask)**2.0, axis=0)).astype(np.uint8)

            # apply baseline defocus to data FPs
            ksim_data_baseline_d = ksim_data_d*cp.exp(-1j*chi_d)

            # apply defocus series to aberrated simulated data, save into final array on host; then, average over FPs
            sim_data_averaged = np.zeros((metadata["numFP"], 640, 640), dtype=np.float32)
            for df, dfw in zip(defoci, defoci_weights):
                df_abb = cp.asarray(pyprismatic.process.calc_defocus_aberration(q, qTheta, q_lambda, df).astype('complex64'))
                sim_data_averaged = sim_data_averaged + dfw*np.abs(fast_apply_aberration(ksim_data_baseline_d, df_abb))**2.0

            sim_data_averaged = np.mean(sim_data_averaged, axis=0)

            # apply poisson noise
            noisy_sim_data = pyprismatic.process.apply_poisson_noise_HRTEM(sim_data_averaged,
                                                                            (128.0, 128.0),
                                                                            dose=dose,
                                                                            normalize=False).astype(np.uint8)

            # apply median filtering
            noisy_sim_data = median_filter(noisy_sim_data, size=3, mode="wrap")

            # choose a random corner in first 128x128 pixels to start 512x512 crop region
            s = rng.integers(0,128, 2)

            # save files
            out_folder = get_new_folder(base="dataset_HQ/img")
            np.save(out_folder.joinpath("train.npy"), noisy_sim_data[s[0]:s[0]+512, s[1]:s[1]+512])
            np.save(out_folder.joinpath("mask.npy"), mask[s[0]:s[0]+512, s[1]:s[1]+512])

            metadata["C1"] = defocus
            metadata["dose"] = dose
            metadata["median_filter_size"] = 3
            metadata["mask_guassian_filter_size"] = 5.0
            metadata["focal_spread"] = defocus_sigma
            metadata["C3"] = 0

            write_metadata(metadata, out_folder.joinpath("metadata.json"))

if __name__ == "__main__":
    main()