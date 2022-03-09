import numpy as np
import segmentation_models as sm
import h5py
import pathlib

from scipy.ndimage import median_filter
from tqdm import tqdm
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
    """Preprocessing function to limit rotations to 0, 90, 180, 270io
    
    based on https://stackoverflow.com/a/61304850/10094666
    """
    return np.rot90(image, np.random.choice([0, 1, 2, -1]))

def get_scheduler(factor):

    def scheduler(epoch, lr):
        return lr*factor

    return scheduler

def date_as_tuple(path):
    return tuple([int(x) for x in str(path).split("/")[-1].split("_")[1:]])[::-1]

def get_id_from_date(dt):
    return str(dt[1])+"_" + str(dt[0])

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
    
    X_CdSe, Y_CdSe = load_CdSe_dataset()
    X_Kath, Y_Kath = load_Katherine_dataset()

    cdse_callback = EvaluateDataset(X_CdSe, Y_CdSe, "CdSe")
    kath_callback = EvaluateDataset(X_Kath, Y_Kath, "Kath")

    print("Data loaded and prepared.")
    print("X shape and memory footprint (Gb): (%i, %i, %i, %i), %f" % (*X.shape, X.nbytes/1e9))
    print("Y shape and memory footprint (Gb): (%i, %i, %i, %i), %f" % (*Y.shape, Y.nbytes/1e9))

    # load models and train
    root_dir_base = pathlib.Path("/media/files/segmentation_networks/")
    root_dirs = [root_dir_base.joinpath("small_NP_networks_with_callbacks_2_11_2022"),
                ]

    root_dirs = [folder.joinpath("trained_models") for folder in root_dirs]
    model_folders = [[x for x in listfiles(r) if x.is_dir()] for r in root_dirs]
    model_folders = reduce(lambda x,y: x+y, model_folders)

    for folder in tqdm(model_folders):
        for validation_split in [0.2, 0.4, 0.5, 0.6, 0.8]:
            # prepare data generators
            seed = int(rng.integers(0, 1e6))
            data_gen_args = {"preprocessing_function":orthogonal_rot, 
                            "horizontal_flip":True, 
                            "vertical_flip":True,
                            "validation_split":validation_split}

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

            callbacks_list = [earlyStopping, schedule, cdse_callback, kath_callback]

            #
            print("Pre-training performance: ")
            print("Kate F1: %0.4f, CdSe F1: %0.4f, Katherine F1: %0.4f" % \
                (model_metadata["exp_f1-score"],
                 model_metadata["exp_f1-score_CdSe"],
                 model_metadata["exp_f1-score_katherine"],))

            # perform training
            N_epochs = 50
            history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch, epochs=N_epochs,
                        callbacks = callbacks_list, validation_data=test_generator,
                        validation_steps=validation_steps, verbose=2)

            # save final weights, training history, metadata
            model_metadata["expt_generator_seed"] = seed
            model_metadata["expt_val_split"] = data_gen_args["validation_split"]
            model_metadata["orig_model_ID"] = get_id_from_date(date_as_tuple(folder))

            write_metadata(model_metadata, model_dir.joinpath("metadata.json"))
            h = h5py.File(history_path,'w')
            h_keys = history.history.keys()
            for k in h_keys:
                h.create_dataset(k,data=history.history[k])
            h.close()

            K.clear_session()


if __name__ == "__main__":
    transfer_learn()
