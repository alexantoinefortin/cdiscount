#!/usr/local/bin/python3
"""
Alex-Antoine Fortin
November 23rd 2017
Description
This python script:
1. Import needed modules
2. Defines model (Xception trained on ImageNet)
4. Freezes layers, Allow training on classifier layers only
5. Training
"""
import os, threading, numpy as np
from keras.preprocessing.image import ImageDataGenerator
from time import time
t = imp.load_source('tools', '../code/tools.py')
startTime = time()
TARGET_SIZE = (150, 150)
BATCH_SIZE = 128

#==============================
#Generating and cropping images
#==============================
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
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
MODEL_NAME = 'Xception'
NB_CLASS = 5270
NB_EPOCH = 65
NB_STEPS_PER_EPOCH = int((7026093/BATCH_SIZE)/10) # 1 epoch is 1/10 of the data
NB_VAL_STEPS_PER_EPOCH = int(20000/BATCH_SIZE)
INIT_LR = 0.1
REDUCE_LR_ON_PLATEAU_FACTOR = 0.1 # every 10th epoch, multiply by this number

model = t.define_model(TARGET_SIZE, NB_CLASS)
# Freeze base_model weights
for layer in base_model.layers:
    layer.trainable = False

optimizer = SGD(lr=INIT_LR, decay=0, momentum=0.9, nesterov=True)
#==========
# Callbacks
#==========
cp = t.cp(path_to_save='../models/{}-unfrozen2'.format(MODEL_NAME))

def scheduler(epoch):
    if ((epoch%10 == 0) & (epoch>0)):
        new_lr = max(K.get_value(
                    model.optimizer.lr)*REDUCE_LR_ON_PLATEAU_FACTOR, 0.00001)
        print("New learning rate:{}".format(new_lr))
    else:
        new_lr = K.get_value(model.optimizer.lr)
        print("Old and current learning rate:{}".format(new_lr))
    return new_lr

lr_adjustment = LearningRateScheduler(scheduler) # every 10

#=======================================================
# compile the model
# should be done *after* setting layers to non-trainable
#=======================================================
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

# print(model.summary())
# train the model on the new data for a few epochs
print("NB_EPOCH:{}".format(NB_EPOCH))
print("nb of minibatches to process (train):{}".format(NB_STEPS_PER_EPOCH))
print("nb of minibatches to process (test) :{}".format(NB_VAL_STEPS_PER_EPOCH))
model.fit_generator(
        generator=crop_traingen,
        steps_per_epoch=NB_STEPS_PER_EPOCH,
        epochs=NB_EPOCH,
        workers=12,
        callbacks=[cp, lr_adjustment],
        validation_data=crop_testgen,
        validation_steps=NB_VAL_STEPS_PER_EPOCH)

# at this point, the top layer(s) are well trained and we can start fine-tuning
# convolutional layers from the base_model. We will freeze the bottom N layers
# and train the remaining top layers.
