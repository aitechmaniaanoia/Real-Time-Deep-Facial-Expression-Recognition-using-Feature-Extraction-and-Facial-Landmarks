# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:16:17 2020

@author: Zoe
"""

import os
import argparse
import numpy as np
import keras
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

import config
from dataset import *
from model import *


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required = True, help = "path to output checkpoint directory")
ap.add_argument("-m", "--model", type = str, help = "path to specific model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type = int, default = 0, help = "epoch to restart training at")
args = vars(ap.parse_args())

# prepare dataset
X_train, Y_train = TrainDataGenerator(data_path = config.BASE_PATH, 
                                      size = 48, 
                                      classes = config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("compiling model...")
    model = EmotionVGGNet.build(width = 48, height = 48, depth = 1,
        classes = config.NUM_CLASSES)
    # opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    opt = Adam(lr = 1e-3)
    model.compile(loss = "categorical_crossentropy", optimizer = opt,
        metrics = ["accuracy"])
    
# otherwise, load the checkpoint from disk
else:
    print("[INFO] loding {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
    

# train network
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.fit(X_train, Y_train, 
          batch_size=config.BATCH_SIZE, 
          epoch=config.EPOCH, 
          verbose=1,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          Validatiuon_split=0.1)

# model.evaluate()

# model.predict()
