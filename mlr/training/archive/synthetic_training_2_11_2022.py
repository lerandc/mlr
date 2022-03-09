import numpy as np
import pandas as pd
import segmentation_models as sm
import h5py
import pathlib

from scipy.ndimage import median_filter
from tqdm import tqdm
from itertools import product, combinations
from functools import reduce
from ttt.utils import listfiles
from mlr.database.utils import get_new_folder, read_metadata, write_metadata

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

class EvaluateDataset(tf.keras.callbacks.Callback):

    def __init__(self, X, Y, log_name, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.Y = Y
        self.log_name = log_name

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(self.X, self.Y, batch_size=16)
        metric_names = [self.log_name+ "_" + x.name for x in self.model.metrics]

        for key, val in zip(metric_names, results):
            logs[key] = val

def orthogonal_rot(image):
    """Preprocessing function to limit rotations to 0, 90, 180, 270
    
    based on https://stackoverflow.com/a/61304850/10094666
    """
    return np.rot90(image, np.random.choice([0, 1, 2, -1]))

def get_scheduler(factor):

    def scheduler(epoch, lr):
        return lr*factor

    return scheduler

def load_Kate_Au_dataset():
    # load data
    expt_data_path = "/media/files/nanoparticle_expt_data/kate_zenodo/Au_Bal_MedFilt_cutimages_20190726.h5"
    expt_mask_path = "/media/files/nanoparticle_expt_data/kate_zenodo/Au_Bal_unFilt_cutimages_20190423_maps.h5"

    f_tmp = h5py.File(expt_data_path, "r")
    X = np.array(f_tmp["images"]).astype(np.float32)
    f_tmp.close()

    f_tmp = h5py.File(expt_mask_path, "r")
    Y = np.array(f_tmp["maps"]).astype(np.float32)
    f_tmp.close()
    return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)

def load_CdSe_dataset():
    expt_data_path = "/media/files/nanoparticle_expt_data/kate_zenodo/Bal_MedFilt_CdSeRelabel512Images_20190726.h5"
    expt_mask_path = "/media/files/nanoparticle_expt_data/kate_zenodo/Bal_unFilt_CdSeRelabel512Images_20190724_maps.h5"

    f_tmp = h5py.File(expt_data_path, "r")
    X = np.array(f_tmp["images"]).astype(np.float32)
    f_tmp.close()

    f_tmp = h5py.File(expt_mask_path, "r")
    Y = np.array(f_tmp["maps"]).astype(np.float32)
    f_tmp.close()

    return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)

def load_Katherine_dataset():
    expt_data_path = "/media/files/nanoparticle_expt_data/ksytwu/Au_2p2nm/Au_2p2nm_330kx_423e_raw_UTC_Team05_Images.h5"
    expt_mask_path = "/media/files/nanoparticle_expt_data/ksytwu/Au_2p2nm/Au_2p2nm_330kx_423e_raw_UTC_Team05_Labels.h5"

    f_tmp = h5py.File(expt_data_path, "r")
    X_tmp = np.array(f_tmp["images"]).astype(np.float32)[:,:,:,None]
    f_tmp.close()

    f_tmp = h5py.File(expt_mask_path, "r")
    Y_tmp = np.array(f_tmp["labels"]).astype(np.float32)
    f_tmp.close()
    
    X = np.zeros_like(X_tmp)
    Y = np.zeros(shape=(*Y_tmp.shape,2), dtype=np.float32)
    Y[:,:,:,0] = 1-Y_tmp
    Y[:,:,:,1] = Y_tmp

    for i in range(X_tmp.shape[0]):
        Xt = median_filter(X_tmp[i,:,:,0], size=3, mode="reflect")
        Xt_mm = (Xt.min(), Xt.max())
        X[i,:,:,0] = (Xt-Xt_mm[0])/(Xt_mm[1]-Xt_mm[0])

    return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)

def set_in_ranges(dataframe, key, ranges):

    filters = []
    for r in ranges:
        filters.append((dataframe[key] >= r[0]) &  (dataframe[key] <= r[1]))

    return dataframe[reduce(lambda x,y: x | y, filters)]

def fraction_in_range(dataframe, key, ranges, return_set=False):

    N_tot = float(len(dataframe))


    if return_set:
        out_dataframe = set_in_ranges(dataframe, key, ranges)
        return out_dataframe, float(len(out_dataframe))/N_tot
    else:
        N = float(len(set_in_ranges(dataframe, key, ranges)))
        return N/N_tot

def reduce_dataframe(dataframe, strategy=None, **kwargs):

    unique_ids = dataframe["particle_ID"].drop_duplicates().sample(n=kwargs["N_structures"], random_state=kwargs["seed"])

    dataframe = dataframe[dataframe["particle_ID"].isin(unique_ids)]
    if strategy=="2":
        dataframe, meta_keys = reduce_defocus_2(dataframe, **kwargs)
    elif strategy=="2b":
        dataframe, meta_keys = reduce_defocus_2b(dataframe, **kwargs)
    else:
        dataframe, meta_keys = dataframe.sample(frac=kwargs["N_defocus"]/11.0, random_state=kwargs["seed"]+1000)

    # further sample so that we can remove the effect of pure data amount
    if len(dataframe) > 1024: 
        dataframe = dataframe.sample(n=1024, 
                                    random_state=kwargs["seed"]+2000,
                                    weights="sampling_weights" if "sampling_weights" in dataframe.columns else None)

    return dataframe, meta_keys

def reduce_defocus_2(dataframe, cv_sets=[None], N=1, return_sets=False, **kwargs):

    """    
    ## TODO: Should add a function to MLR to set up an RNG and seed.
    f set_up_rng(kwargs):
        if rng exists->return rng, else set up default
        if seed exists->return seed, else draw an integer
        if need to seed RNG, use the previous seed as a global seed, 
        seed the RNG appropriately, and draw a new seed; return all three
    """
    
    if "rng" in kwargs.keys():
        l_rng = kwargs["rng"]
    else:
        l_rng = np.random.default_rng()

    target_sets = l_rng.choice(list(cv_sets.keys()), size=N, replace=False)
    target_ranges = [cv_sets[x] for x in target_sets]
    meta_keys = list(target_sets)

    if return_sets:
        return target_ranges
    else:
        return set_in_ranges(dataframe, key="C1", ranges=target_ranges), meta_keys

def reduce_defocus_2b(dataframe, cv_super_sets=None, return_sets=False, **kwargs):

    """    
    ## TODO: Should add a function to MLR to set up an RNG and seed.
    f set_up_rng(kwargs):
        if rng exists->return rng, else set up default
        if seed exists->return seed, else draw an integer
        if need to seed RNG, use the previous seed as a global seed, 
        seed the RNG appropriately, and draw a new seed; return all three
    """
    
    if "rng" in kwargs.keys():
        l_rng = kwargs["rng"]
    else:
        l_rng = np.random.default_rng()

    target_ranges = []
    meta_keys = []
    for cv_set in cv_super_sets:
        target_sets = l_rng.choice(list(cv_set.keys()))
        meta_keys.extend([target_sets])
        target_ranges.extend([cv_set[x] for x in target_sets])

    if return_sets:
        return target_ranges
    else:
        return set_in_ranges(dataframe, key="C1", ranges=target_ranges), meta_keys

def main():
    rng = np.random.default_rng()

    #params include
    batch_size = 16
    N_defocus = [7, 9, 11] #[3, 5, 7, 9, 11]
    N_structures = [128, 192, 256, 384, 512]
    backbone = ["resnet18"]
    alpha_0 = [0.01] #starting learning rate
    schedule = [(0.8, "every_epoch")]

    param_list = list(product(*(backbone, alpha_0, schedule, N_defocus, N_structures)))
    param_list = [x for x in param_list if x[3]*x[4] >= 1024]
    for p in tqdm(param_list):
        if p[3] < 7:
            N_draws = 10
        else:
            N_draws = 5
        
        for i in range(N_draws):
            seed = int(rng.integers(0, 1e6))
            params = {"batch_size":batch_size,
                    "target_dose":400,
                    "backbone":p[0],
                    "alpha_0":p[1],
                    "schedule":p[2],
                    "N_defocus":p[3],
                    "N_structures":p[4],
                    "seed":seed
                    }
            try:
                run_training(**params)
                K.clear_session()
            except:
                print("Network training failed for following parameters: ", params)
                print("Training failed on iteration %i" % i)



def run_training(**kwargs):

    # load data
    root_folder = "/media/files/simulated_databases/defected_Au_np_processed_images_thermal_1_1_5_2022/dataset"

    folders = [x for x in listfiles(root_folder) if x.is_dir()]

    metadata_list = []
    for f in folders:
        dataX_path = f.joinpath("train.npy")
        dataY_path = f.joinpath("mask.npy")
        meta_dict = read_metadata(f.joinpath("metadata.json"))
        meta_dict["dataX_path"] = dataX_path
        meta_dict["dataY_path"] = dataY_path
        metadata_list.append(meta_dict)

    df = pd.DataFrame(metadata_list)
    ## filter dataframe by unique particles and then by defocus

    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    cv_centers = {x:y for x,y in zip(letters,
                                        np.linspace(-500,500,11))
                }

    cv_ranges = {}
    target_frac = 1.0/11.0
    for key, val in cv_centers.items():

        rad = 50.0
        frac = 0.0
        while(frac < target_frac):
            rad += 1.0
            frac = fraction_in_range(df, key="C1", ranges=[(val-rad, val+rad)])

        cv_ranges[key] = (val-rad, val+rad)
        
    cv_super_3 = [{x:cv_ranges[x] for x in ["A", "B", "C", "D"]},
                {x:cv_ranges[x] for x in ["E", "F", "G"]},
                {x:cv_ranges[x] for x in ["H", "I", "J", "K"]},]

    cv_super_5 = [{x:cv_ranges[x] for x in ["A", "B"]},
                {x:cv_ranges[x] for x in ["C", "D"]},
                {x:cv_ranges[x] for x in ["E", "F", "G"]},
                {x:cv_ranges[x] for x in ["H", "I"]},
                {x:cv_ranges[x] for x in ["J", "K"]},]

    if kwargs["N_defocus"] == 3:
        df, focal_ranges = reduce_dataframe(df, seed=kwargs["seed"], cv_super_sets=cv_super_3, N_structures=kwargs["N_structures"], strategy="2b")
    elif kwargs["N_defocus"] == 5:
        df, focal_ranges = reduce_dataframe(df, seed=kwargs["seed"], cv_super_sets=cv_super_5, N_structures=kwargs["N_structures"], strategy="2b")
    else:
        df, focal_ranges = reduce_dataframe(df, seed=kwargs["seed"], cv_sets=cv_ranges, N=kwargs["N_defocus"], N_structures=kwargs["N_structures"], strategy="2")

    kwargs["focal_ranges"] = focal_ranges
    
    X = np.array([np.load(f) for f in df["dataX_path"]])
    X_min = np.min(X, axis=(1,2))[:,None,None]
    X_max = np.max(X, axis=(1,2))[:,None,None]
    X = (X-X_min) / (X_max - X_min)
    X = np.expand_dims(X,axis=3)

    Y = np.array([np.load(f) for f in df["dataY_path"]]).astype(np.float32)
    Y = np.expand_dims(Y,axis=3)
    Y = np.array(np.concatenate((np.abs(Y-1), Y), axis=3))

    print("Loading auxiliary datasets.")
    X_Kate, Y_Kate = load_Kate_Au_dataset()
    X_CdSe, Y_CdSe = load_CdSe_dataset()
    X_Kath, Y_Kath = load_Katherine_dataset()

    kate_callback = EvaluateDataset(X_Kate, Y_Kate, "Kate")
    cdse_callback = EvaluateDataset(X_CdSe, Y_CdSe, "CdSe")
    kath_callback = EvaluateDataset(X_Kath, Y_Kath, "Kath")

    # prepare data generators
    seed = kwargs["seed"]
    data_gen_args = {"preprocessing_function":orthogonal_rot, 
                    "horizontal_flip":True, 
                    "vertical_flip":True,
                    "validation_split":0.25}

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(X, augment=True, seed=seed)
    mask_datagen.fit(Y, augment=True, seed=seed)


    batch_size = kwargs["batch_size"]
    
    # should I use a factor of 8 for the augmentation size increase? probably not
    steps_per_epoch = X.shape[0]*(1-data_gen_args["validation_split"])//batch_size 
    validation_steps = X.shape[0]*(data_gen_args["validation_split"])//batch_size 

    image_generator_train = image_datagen.flow(X, batch_size=batch_size, seed=seed, subset='training')
    image_generator_test  = image_datagen.flow(X, batch_size=batch_size, seed=seed, subset='validation')

    mask_generator_train = mask_datagen.flow(Y, batch_size=batch_size, seed=seed, subset='training')
    mask_generator_test  = mask_datagen.flow(Y, batch_size=batch_size, seed=seed, subset='validation')


    train_generator = (pair for pair in zip(image_generator_train, mask_generator_train)) #https://stackoverflow.com/a/65731446/10094666
    test_generator = (pair for pair in zip(image_generator_test, mask_generator_test))

    # prepare checkpointing and monitoring
    backbone = kwargs["backbone"]
    model_dir = get_new_folder(base="trained_models/unet")
    weights_path = model_dir.joinpath("sm_unet_noPretrainWeights_" + backbone + "_weights.h5")
    history_path = model_dir.joinpath("sm_unet_noPretrainWeights_" + backbone + "_history.h5")

    modelCheckpoint = ModelCheckpoint(weights_path,
                                  monitor = 'val_loss',
                                  save_best_only = True,
                                  mode = 'min',
                                  verbose = 2,
                                  save_weights_only = True)

    earlyStopping = EarlyStopping(monitor='val_loss',
                                patience=5,
                                verbose=2,
                                min_delta = 0.001,
                                mode='min',)

    callbacks_list = [modelCheckpoint, kate_callback, kath_callback, cdse_callback]

    lr_schedule = kwargs["schedule"]
    if lr_schedule[1] == "every_epoch":
        schedule = LearningRateScheduler(get_scheduler(lr_schedule[0]))
    elif lr_schedule[1] == "on_plateau":
        schedule = ReduceLROnPlateau(monitor="val_loss", 
                                    factor=lr_schedule[0],
                                    patience=5,
                                    mode="min"
                                    )

    callbacks_list.append(schedule)

    # load model with vgg16 backbone
    model = sm.Unet(backbone, encoder_weights=None, classes=2, activation='softmax', input_shape=(None, None, 1))
    learning_rate = kwargs["alpha_0"]
    model.compile(
        Adam(learning_rate=learning_rate),
        loss = sm.losses.cce_dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score]
    )

    # train with history
    N_epochs = 25
    history = model.fit(train_generator,
                steps_per_epoch=steps_per_epoch, epochs=N_epochs,
                callbacks = callbacks_list,validation_data=test_generator,
                validation_steps=validation_steps, verbose = 2)

    # save training history, metadata
    write_metadata(kwargs, model_dir.joinpath("metadata.json"))
    h = h5py.File(history_path,'w')
    h_keys = history.history.keys()
    print(h_keys)
    for k in h_keys:
        h.create_dataset(k,data=history.history[k])
    h.close()

if __name__ == "__main__":
    main()
