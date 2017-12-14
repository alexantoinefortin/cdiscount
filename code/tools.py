#!/usr/local/bin/python3
"""
Alex-Antoine Fortin
Wednesday Dec 6, 2017
Description
This python script regroups functions used thorough our code for the cdiscount
challenge
"""
import os, imp, threading, numpy as np
from keras.preprocessing.image import ImageDataGenerator

#===============
#Cropping images
#===============
#@jit
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

#@jit
def center_crop(x, center_crop_size, **kwargs):
    """
    Returns a preprocessed centered crop of an array x
    """
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return preprocess_input(x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh])

#@jit
def random_crop(x, random_crop_size, sync_seed=None):
    """
    Return a preprocessed random crop of an array x
    """
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0])
    rangeh = (h - random_crop_size[1])
    cropped = np.empty(shape=(x.shape[0], random_crop_size[0], random_crop_size[1], 3), dtype=float) #initialize empty array
    for i in range(x.shape[0]):
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        cropped[i,...] = x[i,...][offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1],:]
    return cropped

#==================
#Defining data generator
#==================
def genTrain():
    return ImageDataGenerator(
    featurewise_center=False, # Set input mean to 0 over the dataset, feature-wise.
    samplewise_center=False, # Set each sample mean to 0.
    featurewise_std_normalization=False, # Divide inputs by std of the dataset, feature-wise
    samplewise_std_normalization=False, # Divide each input by its std
    zca_whitening=False, # Apply ZCA whitening.
    zca_epsilon=1e-6, # epsilon for ZCA whitening. Default is 1e-6
    rotation_range=0, # Int. Degree range for random rotations
    width_shift_range=0.05, # Float (fraction of total width). Range for random horizontal shifts.
    height_shift_range=0.05, # Float (fraction of total height). Range for random vertical shifts.
    shear_range=0., # Float. Shear Intensity (Shear angle in counter-clockwise direction as radian
    zoom_range=0., # Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
    channel_shift_range=0., # Float. Range for random channel shifts.
    fill_mode='reflect', #  {"constant", "nearest", "reflect" or "wrap"}. Points outside the boundaries of the input are filled according to the given mode
    cval=0., # Value used for points outside the boundaries when fill_mode = "constant"
    horizontal_flip=True, # Randomly flip inputs horizontally.
    vertical_flip=False, # Randomly flip inputs vertically.
    rescale=None, # If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation)
    preprocessing_function=preprocess_input, # The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.
    data_format='channels_last')

def genTest():
    """
    For testing, we only flip and normalize the image
    """
    return ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=preprocess_input
    )

def traingen(genTrain, BATCH_SIZE):
    return genTrain.flow_from_directory(
        '../input/train', #XXX: hard-coded path
        target_size= (180,180), # default: (256, 256)
        color_mode="rgb", # one of "grayscale", "rbg"
        classes=None, # If not provided, does not return a label.
        batch_size=BATCH_SIZE, # default: 32.
        shuffle=True,
        #seed=1234,
        #save_to_dir='../generated_img', #Path to directory where to save generated pics
        class_mode='categorical' #one of "categorical", "binary", "sparse" or None
        )

def testgen(genTest, BATCH_SIZE):
    return genTest.flow_from_directory(
        '../input/test', #XXX: hard-coded path
        target_size= (180, 180),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

def holdoutgen(genTest, BATCH_SIZE, path='../input/holdout'):
    return genTest.flow_from_directory(
        path,
        target_size= (180, 180),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

#=======================================
# Adding random cropping to a generators
#=======================================
class crop_gen:
    """
    Uses @generator to get 1 batch.
    crop_gen randomly select a crop of size @target_size from an image obtained
    via @generator. It then return the randomly cropped image and its label if
    @with_label.
    """
    def __init__(self, generator,target_size):
        self.lock = threading.Lock()
        self.generator = generator
        self.target_size = target_size
    #
    def __iter__(self):
        return self
    #
    def __next__(self):
        with self.lock:
            batch = next(self.generator)
        return (random_crop(batch[0], self.target_size),batch[1])

#================
#Model definition
#================
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras import backend as K
from keras.applications.xception import Xception as loaded_model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

def define_model(TARGET_SIZE, NB_CLASS):
    """
    @TARGET_SIZE: Tuple. Size of each input images.
    """
    input_layer = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
                        name='InputLayer')
    base_model = loaded_model(
        include_top=False,
        weights='imagenet', # None (random init) or 'imagenet'
        input_tensor=input_layer,
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        pooling=None,
        classes=None # optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
        )
    #==================
    #Model architecture
    #==================
    x = base_model.output
    x = GlobalAveragePooling2D(name='AvgPooling2D_after_base_model')(x)
    x = BatchNormalization( axis=1, center=True, scale=True,
                            name='BN_after_AvgPooling')(x)
    predictions = Dense(NB_CLASS, activation='softmax', name='Dense_out')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def cp(path_to_save):
    """
    Defines model checkpoint
    """
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
    return ModelCheckpoint(filepath=os.path.join(path_to_save,
                        'model-{epoch:03d}-{val_acc:.4f}-{val_loss:.4f}.hdf5'),
                        monitor='val_acc',
                        mode='max', # min: for loss, max: for acc, auto: infer from monitor's value
                        save_best_only=False, # if True, the latest best model according to the quantity monitored will not be overwritten
                        save_weights_only=False, # model.save_weights(filepath) if True else model.save(filepath)
                        period=2 # Perform ModelCheckpoint every period epoch
                        )
