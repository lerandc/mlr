import numpy as np
import imageio
import pathlib
import h5py

import segmentation_models as sm

from ttt.utils import listfiles
from mlr.database.utils import read_metadata, write_metadata
from tqdm import tqdm

from tensorflow.keras import backend as K

def get_stack_from_4K(img):
    return np.stack([img[512*x:512*(x+1), 512*y:512*(y+1)] for x in range(8) for y in range(8)], axis=0)

def evaluate_models():

    expt_data_path = "/media/files/nanoparticle_expt_data/kate_zenodo/Au_Bal_MedFilt_cutimages_20190726.h5"
    expt_mask_path = "/media/files/nanoparticle_expt_data/kate_zenodo/Au_Bal_unFilt_cutimages_20190423_maps.h5"

    f_tmp = h5py.File(expt_data_path, "r")
    X = np.array(f_tmp["images"]).astype(np.float32)
    f_tmp.close()

    f_tmp = h5py.File(expt_mask_path, "r")
    Y = np.array(f_tmp["maps"]).astype(np.float32)
    f_tmp.close()
    
    print("Data loaded and prepared.")
    print("X shape and memory footprint (Gb): (%i, %i, %i, %i), %f" % (*X.shape, X.nbytes/1e9))
    print("Y shape and memory footprint (Gb): (%i, %i, %i, %i), %f" % (*Y.shape, Y.nbytes/1e9))

    # load models
    print("Evaluating models")
    model_folders = [x for x in listfiles("trained_models") if x.is_dir()]
    for folder in tqdm(model_folders):
        model_metadata = read_metadata(folder.joinpath("metadata.json"))

        model = sm.Unet(model_metadata["backbone"], encoder_weights=None, classes=2, activation='softmax', input_shape=(None, None, 1))
        model.compile(loss=sm.losses.cce_dice_loss, metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
        model.load_weights(folder.joinpath("sm_unet_noPretrainWeights_" + model_metadata["backbone"] +"_weights.h5"))

        results = model.evaluate(X, Y)
        # model_metadata["exp_loss"] = results[0]
        # model_metadata["exp_iou"] = results[1]
        # model_metadata["exp_f1-score"] = results[2]

        # write_metadata(model_metadata, folder.joinpath("metadata.json"))
        K.clear_session()


if __name__ == "__main__":
    evaluate_models()