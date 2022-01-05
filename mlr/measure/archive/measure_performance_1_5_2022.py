import numpy as np
import imageio
import pathlib

import segmentation_models as sm

from ttt.utils import listfiles
from mlr.database.utils import read_metadata, write_metadata
from tqdm import tqdm

from tensorflow.keras import backend as K

def get_stack_from_4K(img):
    return np.stack([img[512*x:512*(x+1), 512*y:512*(y+1)] for x in range(8) for y in range(8)], axis=0)

def evaluate_models():
    root_train_folder = pathlib.Path("/media/files/nanoparticle_expt_data/20190304_Au/train/")
    root_mask_folder = pathlib.Path("/media/files/nanoparticle_expt_data/20190304_Au/masks/")

    ## load all data and prepare
    X_list = []
    Y_list = []

    print("Loading data.")

    for i in tqdm(list(range(1,7)) + list(range(8,14))):
        train_filename = "2019_AuNP_490kX_trial%0.2i.png" % i
        mask_filename = "Label_%i.png" % i
        X = np.asarray(imageio.imread(root_train_folder.joinpath(train_filename))).astype(np.float32)
        Y = np.asarray(imageio.imread(root_mask_folder.joinpath(mask_filename))).astype(np.float32)
        X = get_stack_from_4K(np.mean(X[:,:,:3], axis=2)) #average over RGB channels
        Y = get_stack_from_4K(Y)
        X_min = np.min(X, axis=(1,2))[:,None,None]
        X_max = np.max(X, axis=(1,2))[:,None,None]
        X = (X-X_min) / (X_max - X_min)
        X = np.expand_dims(X,axis=3)

        Y = np.expand_dims(Y,axis=3)
        Y = np.array(np.concatenate((np.abs(Y-1), Y), axis=3))
        
        X_list.append(X)
        Y_list.append(Y)
        
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    particle_filter = np.sum(Y[:,:,:,1], axis=(1,2)) > 0

    print(X.dtype)
    print(Y.dtype)

    del X_list
    del Y_list

    # load models
    print("Evaluating models")
    model_folders = [x for x in listfiles("trained_models") if x.is_dir()]
    for folder in tqdm(model_folders):
        model_metadata = read_metadata(folder.joinpath("metadata.json"))

        model = sm.Unet(model_metadata["backbone"], encoder_weights=None, classes=2, activation='softmax', input_shape=(None, None, 1))
        model.compile(loss=sm.losses.cce_dice_loss, metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
        model.load_weights(folder.joinpath("sm_unet_noPretrainWeights_" + model_metadata["backbone"] +"_weights.h5"))

        # results = model.evaluate(X, Y)
        # model_metadata["exp_loss"] = results[0]
        # model_metadata["exp_iou"] = results[1]
        # model_metadata["exp_f1-score"] = results[2]

        results = model.evaluate(X[particle_filter,:,:,:], Y[particle_filter,:,:,:])
        # model_metadata["exp_filtered_loss"] = results[0]
        # model_metadata["exp_filtered_iou"] = results[1]
        # model_metadata["exp_filtered_f1-score"] = results[2]

        # write_metadata(model_metadata, folder.joinpath("metadata.json"))
        K.clear_session()


if __name__ == "__main__":
    evaluate_models()