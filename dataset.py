# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:07:37 2020

@author: Zoe
"""

import numpy as np
import random
import math
import os
from os.path import join
import cv2

import matplotlib.pyplot as plt


def illumination_normalize(img):
    hh, ww = img.shape[:2]
    max_ = max(hh, ww)
    
    # illumination normalize
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # separate channels
    y, cr, cb = cv2.split(ycrcb)
    
    # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
    # account for size of input vs 300
    sigma = int(5 * max_ / 300)
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)
    
    # subtract background from Y channel
    y = (y - gaussian + 100)
    
    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])
    
    #convert to BGR
    output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return output


def TrainDataGenerator(data_path, size, classes):
    # images
    # labels
    images_dir = join(data_path, 'images')
    labels_dir = join(data_path, 'labels')
    
    data_files = os.listdir(images_dir)
    random.shuffle(data_files)
    
    X_train = [] # grayscale
    Y_train = [] # one-hot vector
    
    for data_file in data_files[:20]:
        
        img_path = os.path.join(images_dir, data_file)
        label_path = os.path.join(labels_dir, data_file[:-4]+'.txt')
        
        img = cv2.imread(img_path)
        f = open(label_path, "r")
        label = f.read()
        label = label.strip().split()
        
        for i in range(int(len(label)/5)):
            # crop face area
            face = img[int(label[i*5+1]):int(label[i*5+4]), int(label[i*5+2]):int(label[i*5+3]),:]
            
            # illumination normalization
            #illu_norm_face = illumination_normalize(face)
            
            # resize
            face = cv2.resize(face, (size, size))
            
            # convert to grayscale
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # normalized
            norm_face = gray/255
            
            # label
            one_hot_label = [0]*classes
            face_label = int(label[i*5+0])

            # convert label to one-hot
            one_hot_label[face_label] = 1
            
            # save X and Y
            X_train.append(norm_face)
            Y_train.append(one_hot_label)

    return X_train, Y_train

        