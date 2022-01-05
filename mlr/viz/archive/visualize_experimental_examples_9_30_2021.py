import segmentation_models as sm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import pathlib
import h5py
import pathlib
import seaborn as sns
import imageio
import re

from tqdm import tqdm
from ttt.utils import listfiles
from mlr.database.utils import read_metadata
from tensorflow.keras import backend as K

sns.set_style("white")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def get_stack_from_4K(img):
    return np.stack([img[512*x:512*(x+1), 512*y:512*(y+1)] for x in range(8) for y in range(8)], axis=0)

def get_4K_from_stack(stack):
    img = np.zeros((4096, 4096), dtype=stack.dtype)
    
    for i in range(8):
        for j in range(8):
            img[512*i:512*(i+1), 512*j:512*(j+1)] = stack[8*i+j, :, :]
            
    return img

def read_training_history(path):
    history_file = h5py.File(path,'r')
    history_dict = {x:np.array(y) for x, y in history_file.items()} 
    history_file.close()
    return history_dict

def main():

    root_train_folder = pathlib.Path("/media/files/nanoparticle_expt_data/20190304_Au/train/")
    root_mask_folder = pathlib.Path("/media/files/nanoparticle_expt_data/20190304_Au/masks/")

    ## load all data and prepare
    X_list = []
    Y_list = []
    X_orig = []
    Y_orig = []

    print("Loading data.")

    for i in tqdm(list(range(1,7)) + list(range(8,14))):
        train_filename = "2019_AuNP_490kX_trial%0.2i.png" % i
        mask_filename = "Label_%i.png" % i
        X = np.asarray(imageio.imread(root_train_folder.joinpath(train_filename))).astype(np.float32)
        Y = np.asarray(imageio.imread(root_mask_folder.joinpath(mask_filename))).astype(np.float32)
        
        X_orig.append(np.mean(X[:,:,:3],axis=2))
        Y_orig.append(Y)
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

    del X_list
    del Y_list

    ## get list of models
    model_folders = [x for x in listfiles("trained_models") if x.is_dir()]
    metadata_list = []

    for folder in model_folders:
        model_metadata = read_metadata(folder.joinpath("metadata.json"))
        
        history_path = "sm_unet_noPretrainWeights_" + model_metadata["backbone"]+"_history.h5"
        history_dict = read_training_history(folder.joinpath(history_path))
        model_metadata.update(history_dict)
        
        model_id = re.search("[0-9]+_[0-9]+", str(folder).rsplit("/")[-1])[0]
        model_metadata["ID"] = model_id
        metadata_list.append(model_metadata)
        
        model_metadata["final_loss"] = history_dict["loss"][-1]
        model_metadata["final_val_loss"] = history_dict["val_loss"][-1]
        
    df = pd.DataFrame(metadata_list)

    q_list = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    metric_list = ["exp_filtered_loss"]
    for m in metric_list:
        for q in tqdm(q_list):
            cur_ID  = df[np.abs(df[m] - df.quantile(q=q, interpolation="nearest")[m]) < 0.0001]["ID"].iloc[0]
            viz_model_performance(cur_ID, q, m, X, Y, particle_filter)
            K.clear_session()

def viz_model_performance(ID, q, m, X, Y, particle_filter):
    model_folder = pathlib.Path("trained_models/unet_"+ID)
    model_metadata = read_metadata(model_folder.joinpath("metadata.json"))
    model = sm.Unet(model_metadata["backbone"], encoder_weights=None, classes=2, activation='softmax', input_shape=(None, None, 1))
    model.compile(loss=sm.losses.cce_dice_loss)
    model.load_weights(model_folder.joinpath("sm_unet_noPretrainWeights_" + model_metadata["backbone"] +"_weights.h5"))

    idx_list = np.arange(particle_filter.shape[0])[particle_filter]
    losses = []
    for i in idx_list:
        tmp_loss = model.evaluate(np.expand_dims(X[i,:,:,:],axis=0),np.expand_dims(Y[i,:,:,:], axis=0), verbose=0)
        losses.append(tmp_loss)

    loss_indices = np.argsort(losses)
    loss_vals = np.sort(losses)

    scale = 5
    N_rows = 5
    N_cols = 9

    for Ni, quality in zip([0, 7, 15], ["best", "med", "bad"]):
        fig, axes = plt.subplots(N_rows, N_cols, figsize=(scale*N_cols,scale*N_rows), facecolor="w")

        N = Ni*(2*N_rows)

        for j in range(N_cols//3):
            for i in range(N_rows):
                N_t = idx_list[loss_indices[N+i+8*j]]
                test_Y = model.predict(np.expand_dims(X[N_t,:,:,:],axis=0))
                axes[i, 3*j+0].matshow(np.squeeze(X[N_t, :, :, 0]))
                axes[i, 3*j+1].matshow(np.squeeze(Y[N_t, :, :, 1]))
                axes[i, 3*j+2].matshow(test_Y[0,:,:,1])
            
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        q_str = "%0.2i" % int(q*100)
        fig_name = "q_" + q_str + "_" + m + "_" + quality + "_results.pdf"
        plt.savefig(fig_name, dpi=300)


if __name__ == "__main__":
    main()

