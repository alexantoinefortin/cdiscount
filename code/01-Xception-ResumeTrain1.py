#!/usr/local/bin/python3
"""
Alex-Antoine Fortin
Tuesday December 05, 2017
Description
This python script:
1. Import needed modules
2. Defines model as defined in 01-Xception-Model.py
3. Loads weights from .hdf5
4. Freezes fewer layers
5. Resumes training
"""
import os, imp, numpy as np
from time import time
t = imp.load_source('tools', '../code/tools.py')

model_filename_to_load = 'model-018-0.5718-2.1493.hdf5' #XXX
TARGET_SIZE = (150, 150)
BATCH_SIZE = 128

startTime = time()

#==================
#Defining data generator
#==================
genTrain = t.genTrain()
genTest = t.genTest()
traingen = t.traingen(genTrain, BATCH_SIZE)
testgen = t.testgen(genTest, BATCH_SIZE)

crop_traingen = t.crop_gen(traingen, TARGET_SIZE)
crop_testgen = t.crop_gen(testgen, TARGET_SIZE)
print("The process took {}s to initiate.".format(int(time()-startTime)))

#=============================
#Model Specs and Model Fitting
#=============================
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
MODEL_NAME = 'Xception'
NB_CLASS = 5270
NB_EPOCH = 65
NB_STEPS_PER_EPOCH = int((7026093/BATCH_SIZE)/10) # 1 epoch is 1/10 of the data
NB_VAL_STEPS_PER_EPOCH = int(20000/BATCH_SIZE)
INIT_LR = 0.06
REDUCE_LR_ON_PLATEAU_FACTOR = 0.1 # every 10th epoch, multiply by this number

model = t.define_model(TARGET_SIZE, NB_CLASS)
#==================
#Load saved weights
#==================
saved_weights_path = os.path.join(  '../models/Xception-unfrozen/',
                                    model_filename_to_load)
model.load_weights(saved_weights_path)
print("Loaded the weights from model: {}".format(model_filename_to_load))

"""
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
   print(i, layer.name)
"""
# we chose to train the top 2 blocks ([:96]) or top block ([:106]),
# of the middle flow of the XceptionNet
for layer in model.layers[:86]:
   layer.trainable = False
for layer in model.layers[86:]:
   layer.trainable = True

optimizer = SGD(lr=INIT_LR, decay=0, momentum=0.9, nesterov=True)
#=========
#Callbacks
#=========
cp = t.cp(path_to_save='../models/{}-unfrozen2'.format(MODEL_NAME))
def get_lr_adjustment(  REDUCE_LR_ON_PLATEAU_FACTOR, reduce_at_epoch=9,
                        min_lr=0.00001):
    """
    @reduce_at_epoch: Int. Reduces the lr after each reduce_at_epoch epochs
    @REDUCE_LR_ON_PLATEAU_FACTOR: Float. multiply for the lr
    """
    def scheduler(epoch):
        if ((epoch%reduce_at_epoch == 0) & (epoch>0)):
            new_lr = max(K.get_value(
                        model.optimizer.lr)*REDUCE_LR_ON_PLATEAU_FACTOR, min_lr)
            print("New learning rate:{}".format(new_lr))
        else:
            new_lr = K.get_value(model.optimizer.lr)
            print("Old and current learning rate:{}".format(new_lr))
        return new_lr
    #
    return LearningRateScheduler(scheduler) # every 10

lr_adjustment =get_lr_adjustment(REDUCE_LR_ON_PLATEAU_FACTOR, reduce_at_epoch=8)
# train the model on the new data for a few epochs
print("NB_EPOCH:{}".format(NB_EPOCH))
print("nb of minibatches to process (train):{}".format(NB_STEPS_PER_EPOCH))
print("nb of minibatches to process (test) :{}".format(NB_VAL_STEPS_PER_EPOCH))
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
        generator=crop_traingen,
        steps_per_epoch=NB_STEPS_PER_EPOCH,
        epochs=NB_EPOCH,
        workers=12,
        callbacks=[cp, lr_adjustment],
        validation_data=crop_testgen,
        validation_steps=NB_VAL_STEPS_PER_EPOCH)
