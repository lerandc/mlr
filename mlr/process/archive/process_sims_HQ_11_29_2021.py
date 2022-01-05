import numpy as np
import py4DSTEM
import pyprismatic
import cupy as cp
from pyprismatic.process import calc_aberration

from scipy.ndimage import median_filter, gaussian_filter
from mlr.database.utils import get_new_folder, write_metadata, read_metadata
from mlr.sampling.distribution import sample_normal
from ttt.utils import listfiles
from tqdm import tqdm
from cupy.fft import fft2 as cupy_fft2, ifft2 as cupy_ifft2
from itertools import product

def get_final_mask_int(mask):
    final_mask = np.abs(mask-1.0)
    final_mask = final_mask > np.var(final_mask) #sim_data are normalized arund mean=1 by definition; finding pixels which deviate
    final_mask = gaussian_filter(final_mask.astype(np.float32), 5.0) > 1e-1 #blur out atom spacing to get smooth mask
    return mask

def get_final_mask_phase(mask):

    tmp_mask = np.zeros((mask.shape[1], mask.shape[2]))

    for fp in mask:
        phase = np.angle(fp)
        tmp_mask += (phase > np.pi/12)*(1/mask.shape[0])

    final_mask = tmp_mask > 0.5
    final_mask = gaussian_filter(final_mask.astype(np.float32), 5.0) > 1e-1
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

    base_dir = "/media/files/simulated_databases/defected_Au_np_sims_11_22_2021/simulations/"

    # process base case simulations
    folders = [x for x in listfiles(base_dir) if x.is_dir()]

    # sort folders by creation ID
    folders.sort(key=lambda x: int(str(x).rsplit("_", 2)[-2]))

    rng = np.random.default_rng()

    defocus_sigma = 15
    defoci = np.linspace(-(3*defocus_sigma), 3*defocus_sigma+1, 25)
    defoci_weights = pyprismatic.process.rad_weights(defoci, defocus_sigma, np.ones(defoci.shape))

    # enter dimensioned aberrations from team0.5 measurement, in meters and degrees
    # format is m,n, mag, angle
    aberrations = np.array([[2,2, 1.602e-9, -68.5],
                            [3,3, 17.98e-9, -170.7],
                            [3,1, 3*23.9e-9, 29.2],
                            [4,0, -1.913e-6, 0.0],
                            [4,4, 158.3e-9, 89.2],
                            [4,2, 4*42.64e-9, -34.5],
                            [5,5, 3.117e-6, -6.3],
                            [5,3, 4*528.7e-9, -12.7],
                            [5,1, 4*1.678e-6, 104.5],
                            [6,0, 1.048e-3, 0.0],
                            [6,6, 4.821e-6, 60.2]])

    q_lambda = pyprismatic.process.get_lambda(folders[0].joinpath("particle_on_substrate.h5"))

    # covnert angles to radians
    # convert magnitudes to angstroms, then dimensionless
    for a in aberrations:
        a[3] *= np.pi/180
        a[2] *= 1e10
        a[2] *= 2*np.pi/(a[0]*q_lambda)

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
        dose_list = [400] # e- / Ang^2 (~2700 e-/A^2/s at capture rate of ~0.16 seconds)

        # defocus from -50nm to 50nm inclusive, in steps of 10nm, with variance of 2nm
        df_list = np.linspace(-500,500,11, endpoint=True)

        for dose, defocus in tqdm(list(product(*(dose_list, df_list))), leave=False):

            cur_df = sample_normal(rng, defocus, 20)[0] 
            baseline_ab = np.vstack([aberrations, [[2, 0, cur_df* np.pi / q_lambda, 0.0]]])

            # apply defocus to mask FPs; incoherently average and form final mask
            chi_d = cp.asarray(pyprismatic.process.calc_aberration(baseline_ab, q, qTheta, q_lambda).astype(np.complex64)[None,:,:])
            df_mask = fast_apply_aberration(kmask_data_d, chi_d)
            # mask = get_final_mask_int(np.mean(np.abs(df_mask)**2.0, axis=0)).astype(np.float32)
            mask = get_final_mask_phase(df_mask).astype(np.float32)

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
            out_folder = get_new_folder(base="dataset/img")
            np.save(out_folder.joinpath("train.npy"), noisy_sim_data[s[0]:s[0]+512, s[1]:s[1]+512])
            np.save(out_folder.joinpath("mask.npy"), mask[s[0]:s[0]+512, s[1]:s[1]+512])

            baseline_ab_save = [[x for x in y] for y in baseline_ab]
            metadata["C1"] = cur_df
            metadata["applied_aberrations"] = baseline_ab_save
            metadata["dose"] = dose
            metadata["median_filter_size"] = 3
            metadata["mask_guassian_filter_size"] = 5.0
            metadata["focal_spread"] = defocus_sigma

            write_metadata(metadata, out_folder.joinpath("metadata.json"))
        
if __name__ == "__main__":
    main()
