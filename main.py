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
import utils

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required = False, help = "path to output checkpoint directory")
ap.add_argument("-m", "--model", type = str, help = "path to specific model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type = int, default = 0, help = "epoch to restart training at")
args = vars(ap.parse_args())

# prepare dataset
X_train, Y_train, X_validate, Y_validate = TrainDataGenerator(data_path = config.BASE_PATH, 
                                      size = config.SIZE, 
                                      classes = config.NUM_CLASSES,
                                      validate_split = 0.2,
                                      channel = config.NUM_CHANNEL)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("compiling model...")
    model = EmotionVGGNet.build(width = config.SIZE, height = config.SIZE, depth = config.NUM_CHANNEL+1,
        classes = config.NUM_CLASSES)
    # opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    opt = Adam(lr = 1e-3)
    model.compile(loss = "categorical_crossentropy", optimizer = opt,
        metrics=['acc', utils.f1_m, utils.precision_m, utils.recall_m])
    
# otherwise, load the checkpoint from disk
else:
    print("[INFO] loding {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-4)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
    

# train network
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
mcp_save = ModelCheckpoint('output/.mdl_wts.hdf5', save_best_only=True, monitor='val_f1_m', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

hist = model.fit(X_train, Y_train, 
          batch_size=config.BATCH_SIZE, 
          epochs=config.EPOCH, 
          verbose=1,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=(X_validate, Y_validate))

# # print training accuracy
# train_loss = hist.history['loss']
# train_acc = hist.history['accuracy']
# validate_loss = hist.history['val_loss']
# validate_acc = hist.history['val_accuracy']
# epoch = list(range(1,len(train_loss)+1))

# utils.plotting_loss(epoch, train_loss, validate_loss, title='Epoch vs Loss')
# utils.plotting_acc(epoch, train_acc, validate_acc, title='Epoch vs Accuracy')

# evaluate on test data
X_test, Y_test = TestDataGenerator(data_path = config.BASE_PATH, 
                                      size = config.SIZE, 
                                      classes = config.NUM_CLASSES,
                                      channel = config.NUM_CHANNEL)
# load the best mdoel
#model = load_model('output/experiment_1/.mdl_wts.hdf5')
pred = model.predict(X_test)

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test, verbose=0)

from sklearn.metrics import classification_report, precision_recall_fscore_support
print(classification_report(np.argmax(Y_test, axis = 1), np.argmax(pred, axis = 1)))


# # find best validate accuacy and its training accuracy
# index = np.argmax(validate_acc)
# best_validate_acc = validate_acc[index]
# best_training_acc = train_acc[index]

# # check the accuracy in test
# pred = np.argmax(pred, axis = 1)
# test_acc = sum(pred == Y_test)/len(Y_test)

# # error analysis
# ## The Occurrences of each Emotion in Dataset
# utils.count_labels(Y_train, Y_validate)

# # error in test
# utils.predict_analysis(Y_test, pred)





