import numpy as np
import segmentation_models as sm
import h5py
import pathlib

from tqdm import tqdm
from functools import reduce
from ttt.utils import listfiles
from mlr.database.utils import get_new_folder, read_metadata, write_metadata

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from segmentation_models.base import Loss

def orthogonal_rot(image):
    """Preprocessing function to limit rotations to 0, 90, 180, 270
    
    based on https://stackoverflow.com/a/61304850/10094666
    """
    return np.rot90(image, np.random.choice([0, 1, 2, -1]))

def get_scheduler(factor):

    def scheduler(epoch, lr):
        return lr*factor

    return scheduler

class IsingLoss(Loss):

    def __init__(self, class_weights=None, class_indexes=None):
        super().__init__(name='ising_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes

    def __call__(self, gt, pr):
        y = 2*(pr-0.5)
        N = tf.cast(tf.size(y), tf.dtypes.float32)
        H = -1*( tf.reduce_sum(y[:,:-1,:,:]*y[:,1:,:,:]) + tf.reduce_sum(y[:,:,:-1,:]*y[:,:,1:,:]))/N
        return tf.math.exp(H)

class IsingLoss_low_T(Loss):

    def __init__(self, class_weights=None, class_indexes=None):
        super().__init__(name='ising_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes

    def __call__(self, gt, pr):
        y = 2*(tf.math.sigmoid(10*(pr-0.5))-0.5)
        N = tf.cast(tf.size(y), tf.dtypes.float32)
        H = -1*( tf.reduce_sum(y[:,:-1,:,:]*y[:,1:,:,:]) + tf.reduce_sum(y[:,:,:-1,:]*y[:,:,1:,:]))/N
        return tf.math.exp(H/0.5)

def transfer_learn():
    rng = np.random.default_rng()

    # params include
    batch_size = 16

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

    # load models and train
    root_dir_base = pathlib.Path("/media/files/segmentation_networks/")
    root_dirs = [root_dir_base.joinpath("baseline_small_datasets_9_30_2021"),
                root_dir_base.joinpath("baseline_small_datasets_HQ_10_7_2021"),
                root_dir_base.joinpath("ising_loss_training_10_9_2021"),
                root_dir_base.joinpath("ising_loss_10_10_2021")
                ]

    root_dirs = [folder.joinpath("trained_models") for folder in root_dirs]
    model_folders = [[x for x in listfiles(r) if x.is_dir()] for r in root_dirs]
    model_folders = reduce(lambda x,y: x+y, model_folders)

    for folder in tqdm(model_folders[300:]):

        # prepare data generators
        seed = int(rng.integers(0, 1e6))
        data_gen_args = {"preprocessing_function":orthogonal_rot, 
                        "horizontal_flip":True, 
                        "vertical_flip":True,
                        "validation_split":0.5}

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        image_datagen.fit(X, augment=True, seed=seed)
        mask_datagen.fit(Y, augment=True, seed=seed)

        steps_per_epoch = X.shape[0]*(1-data_gen_args["validation_split"])//batch_size 
        validation_steps = X.shape[0]*(data_gen_args["validation_split"])//batch_size 

        image_generator_train = image_datagen.flow(X, batch_size=batch_size, seed=seed, subset='training')
        image_generator_test  = image_datagen.flow(X, batch_size=batch_size, seed=seed, subset='validation')

        mask_generator_train = mask_datagen.flow(Y, batch_size=batch_size, seed=seed, subset='training')
        mask_generator_test  = mask_datagen.flow(Y, batch_size=batch_size, seed=seed, subset='validation')

        train_generator = (pair for pair in zip(image_generator_train, mask_generator_train)) #https://stackoverflow.com/a/65731446/10094666
        test_generator = (pair for pair in zip(image_generator_test, mask_generator_test))

        # load model metadata and get relevant variables
        model_metadata = read_metadata(folder.joinpath("metadata.json"))

        # set up paths
        model_dir = get_new_folder(base="trained_models/unet")
        weights_path = model_dir.joinpath("sm_unet_transferLearnWeights_" + model_metadata["backbone"] + "_weights.h5")
        history_path = model_dir.joinpath("sm_unet_transferLearnWeights_" + model_metadata["backbone"] + "_history.h5")

        # set up model and compile
        model = sm.Unet(model_metadata["backbone"], encoder_weights=None, classes=2, activation='softmax', input_shape=(None, None, 1))
        loss_fn = sm.losses.cce_dice_loss
        if len(model_metadata["loss_function"]) == 2:
            if model_metadata["loss_function"][1] == "ising_loss":
                loss_fn = loss_fn + IsingLoss()
            elif model_metadata["loss_function"][1] == "ising_loss_low_T":
                loss_fn = loss_fn + IsingLoss_low_T()

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

        learning_rate = model_metadata["alpha_0"]/2
        model.compile(Adam(learning_rate=learning_rate), loss=loss_fn, metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
        model.load_weights(folder.joinpath("sm_unet_noPretrainWeights_" + model_metadata["backbone"] +"_weights.h5"))

        lr_schedule = model_metadata["schedule"]
        if lr_schedule[1] == "every_epoch":
            schedule = LearningRateScheduler(get_scheduler(lr_schedule[0]))
        elif lr_schedule[1] == "on_plateau":
            schedule = ReduceLROnPlateau(monitor="loss", 
                                        factor=lr_schedule[0],
                                        patience=5,
                                        mode="min"
                                        )

        callbacks_list = [modelCheckpoint, earlyStopping, schedule]

        # perform training
        N_epochs = 50
        history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch, epochs=N_epochs,
                    callbacks = callbacks_list, validation_data=test_generator,
                    validation_steps=validation_steps, verbose = 2)

        # save final weights, training history, metadata
        model_metadata["expt_generator_seed"] = seed
        model_metadata["expt_val_split"] = data_gen_args["validation_split"]

        write_metadata(model_metadata, model_dir.joinpath("metadata.json"))
        h = h5py.File(history_path,'w')
        h_keys = history.history.keys()
        for k in h_keys:
            h.create_dataset(k,data=history.history[k])
        h.close()

        K.clear_session()


if __name__ == "__main__":
    transfer_learn()