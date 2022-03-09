import numpy as np
import pandas as pd
import segmentation_models as sm
import h5py
import pathlib

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

def orthogonal_rot(image):
    """Preprocessing function to limit rotations to 0, 90, 180, 270
    
    based on https://stackoverflow.com/a/61304850/10094666
    """
    return np.rot90(image, np.random.choice([0, 1, 2, -1]))


def get_scheduler(factor):

    def scheduler(epoch, lr):
        return lr*factor

    return scheduler

def main():
    rng = np.random.default_rng()

    #params include
    batch_size = 16
    backbone = ["resnet18"]
    alpha_0 = [0.01, 0.005, 0.001] #starting learning rate
    schedule = [(0.8, "every_epoch")]
    val_splits = [0.2, 0.4, 0.5, 0.6, 0.8]

    param_list = list(product(*(backbone, alpha_0, schedule, val_splits)))
    for p in tqdm(param_list):
        for i in range(5):
            seed = int(rng.integers(0, 1e6))
            params = {"batch_size":batch_size,
                    "target_dose":400,
                    "backbone":p[0],
                    "alpha_0":p[1],
                    "schedule":p[2],
                    "validation_split":p[3],
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

    # prepare data generators
    seed = kwargs["seed"]
    data_gen_args = {"preprocessing_function":orthogonal_rot, 
                    "horizontal_flip":True, 
                    "vertical_flip":True,
                    "validation_split":kwargs["validation_split"]}

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

    earlyStopping = EarlyStopping(monitor='loss',
                                patience=10,
                                verbose=2,
                                min_delta = 0.001,
                                mode='min',)


    lr_schedule = kwargs["schedule"]
    if lr_schedule[1] == "every_epoch":
        schedule = LearningRateScheduler(get_scheduler(lr_schedule[0]))
    elif lr_schedule[1] == "on_plateau":
        schedule = ReduceLROnPlateau(monitor="val_loss", 
                                    factor=lr_schedule[0],
                                    patience=5,
                                    mode="min"
                                    )

    callbacks_list = [earlyStopping, schedule]

    # load model with vgg16 backbone
    model = sm.Unet(backbone, encoder_weights=None, classes=2, activation='softmax', input_shape=(None, None, 1))
    learning_rate = kwargs["alpha_0"]
    model.compile(
        Adam(learning_rate=learning_rate),
        loss = sm.losses.cce_dice_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score]
    )

    # train with history
    N_epochs = 50
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
