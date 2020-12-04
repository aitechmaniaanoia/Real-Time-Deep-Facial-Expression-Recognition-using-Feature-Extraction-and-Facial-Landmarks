# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:41:54 2020

@author: Zoe
"""

from os import path

# define the base path to the emotion dataset
BASE_PATH = "./dataset"

# define the number of classes
NUM_CLASSES = 7
NUM_CHANNEL = 1
# NUM_CLASSES = 6 # use this one if ignore the "disgust" class

# define the size of input data
SIZE = 48

# define the path to output training, validation, and testing HDF5 files
# TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
# VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
# TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

# define the batch size and epoch
BATCH_SIZE = 128
EPOCH = 100

# define the path to where output logs will be stored
OUTPUT_PATH = "output"

