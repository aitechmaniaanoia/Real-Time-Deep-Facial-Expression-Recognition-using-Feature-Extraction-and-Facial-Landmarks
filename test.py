# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:47:25 2020

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
import utils

# compile model
model = EmotionVGGNet.build(width = config.SIZE, height = config.SIZE, depth = config.NUM_CHANNEL,
    classes = config.NUM_CLASSES)
# opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
# opt = Adam(lr = 1e-3)
# model.compile(loss = "categorical_crossentropy", optimizer = opt,
#     metrics=['acc', utils.f1_m, utils.precision_m, utils.recall_m])

# load model weight
model = load_model('output/experiment_2/.mdl_wts.hdf5')#, 
                   #custom_objects={'f1_m':utils.f1_m, 'precision_m': utils.precision_m, 'recall_m': utils.recall_m})

# load dataset
X_test, Y_test = TestDataGenerator(data_path = config.BASE_PATH, 
                                      size = config.SIZE, 
                                      classes = config.NUM_CLASSES,
                                      channel = config.NUM_CHANNEL)
# predict test dataset
pred = model.predict(X_test)

#loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test, verbose=0)

from sklearn.metrics import classification_report, precision_recall_fscore_support
print(classification_report(np.argmax(Y_test, axis = 1), np.argmax(pred, axis = 1)))



