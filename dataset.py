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
import urllib.request as urlreq

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
    if sigma == 0:
        sigma = 1
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)
    
    # subtract background from Y channel
    y = (y - gaussian + 100)
    
    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])
    
    #convert to BGR
    output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return output


def face_processing_baseline(face, size, face_label, classes, channel, landmark_detector):    
    # grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    if channel == 1:
        face = gray
    elif channel == 3:
        # illumination normalization
        face = illumination_normalize(face)
    
    # resize
    face = cv2.resize(face, (size, size))
    
    # facial landmark
    bbox = np.array([[0,0,gray.shape[0],gray.shape[1]]])
    
    # Detect landmarks on "gray"
    _, landmarks = landmark_detector.fit(gray, bbox)
    
    landmarks = landmarks[0][0]
    
    # remove jaw index 0 to 17 
    landmarks = landmarks[18:,:]
    
    idx1 = np.where(landmarks[:,0] > face.shape[1])[0]
    idx2 = np.where(landmarks[:,1] > face.shape[0])[0]
    idx = np.concatenate((idx1, idx2), axis=0)
    if len(idx) > 0:
        # remove landmark
        landmarks = np.delete(landmarks, idx, axis=0)
    
    landmarks_layer = np.zeros(face.shape[:2])
    landmarks_layer[landmarks[:,1].astype(int), landmarks[:,0].astype(int)] = 1
    
    # blur image
    #face = cv2.GaussianBlur(face,(5,5),0) ####################
    
    # normalize
    norm_face = face/255 #################
    # zero-center normalize
    # norm_face = (face - face.mean()) / face.std() ##########################
    
    norm_face = norm_face.reshape((size, size, channel))
    
    face_out = np.zeros((size, size, channel+1))
    face_out[:,:,:-1] = norm_face
    face_out[:,:,-1] = landmarks_layer
    
    #face_out = norm_face
    
    # label convert one-hot
    label_out = [0]*classes

    # convert label to one-hot
    label_out[face_label] = 1
    
    return face_out, label_out


def augmentation(face, augments):
    if augments == 1:
        # flip
        face_out = cv2.flip(face, 1)
        
    elif augments == 2:
        # v shear
        (h, w) = face.shape[:2]
        M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
        M2[0,2] = -M2[0,1] * w/2
        M2[1,2] = -M2[1,0] * h/2
        face_out = cv2.warpAffine(face, M2, (w, h))
        
    elif augments == 3:
        # rotate + shear
        (h, w) = face.shape[:2]
        M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
        M2[0,2] = -M2[0,1] * w/2
        M2[1,2] = -M2[1,0] * h/2
        face = cv2.warpAffine(face, M2, (w, h))
        
        # rotate - 15
        (h, w) = face.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle=-15, scale=1.0)
        face_out = cv2.warpAffine(face, M, (w, h))
        
    elif augments == 4:
        # rotate + 15
        (h, w) = face.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle=15, scale=1.0)
        face_out = cv2.warpAffine(face, M, (w, h))
        
    elif augments == 5:
        # rotate - 15
        (h, w) = face.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle=-15, scale=1.0)
        face_out = cv2.warpAffine(face, M, (w, h))
        
    elif augments == 6:
        # flip + rotate
        face = cv2.flip(face, 1)
        (h, w) = face.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle=10, scale=1.0)
        face_out = cv2.warpAffine(face, M, (w, h))
        
    else:
        # flip + shear
        face = cv2.flip(face, 1)
        (h, w) = face.shape[:2]
        M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
        M2[0,2] = -M2[0,1] * w/2
        M2[1,2] = -M2[1,0] * h/2
        face_out = cv2.warpAffine(face, M2, (w, h))
        
    return face_out

def loadLBFmodel():
    # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "lbfmodel.yaml"
    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
        print("File exists")
    else:
        # download picture from url and save locally as lbfmodel.yaml, < 54MB
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("File downloaded")
    
    # create an instance of the Facial landmark Detector with the model
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    
    return landmark_detector


def TrainDataGenerator(data_path, size, classes, validate_split, channel):

    images_dir = join(data_path, 'images')
    labels_dir = join(data_path, 'labels')
    
    data_files = os.listdir(images_dir)
    random.shuffle(data_files)
    
    X_train = [] # grayscale
    Y_train = [] # one-hot vector
    
    landmark_detector = loadLBFmodel()
    
    for data_file in data_files:
        
        img_path = os.path.join(images_dir, data_file)
        label_path = os.path.join(labels_dir, data_file[:-4]+'.txt')
        
        img = cv2.imread(img_path)
        f = open(label_path, "r")
        label = f.read()
        label = label.strip().split()
        
        for i in range(int(len(label)/5)):
            # crop face area
            face = img[int(label[i*5+1]):int(label[i*5+4]), int(label[i*5+2]):int(label[i*5+3]),:]
            
            face_label = int(label[i*5+0])
            face_out, label_out = face_processing_baseline(face, size, face_label, classes, channel, landmark_detector)
            
            # save X and Y
            X_train.append(face_out)
            Y_train.append(label_out)
            
            # class-based argumentation (face / face_label)
            if face_label == 0: # angry 5
                augments = 5
            elif face_label == 1: # disgust 6
                augments = 6
            elif face_label == 2: # fear 7
                augments = 7
            elif face_label == 4: # sad 3
                augments = 3
            elif face_label == 5: # surprise 3
                augments = 3
            else:
                if_aug = random.randint(0, 1) # happy and natural 0 or 1
                if if_aug == 0:
                    augments = 0 
                else:
                    augments = 1
                
            while augments > 0:
                face_new = augmentation(face, augments)
                face_out, label_out = face_processing_baseline(face_new, size, face_label, classes, channel, landmark_detector)
                X_train.append(face_out)
                Y_train.append(label_out)
                augments -=1

    # shuffle X and Y together 
    combine = list(zip(X_train, Y_train))
    random.shuffle(combine)
    X_train, Y_train = zip(*combine)
    
    if validate_split:
        val_len = int(len(X_train) - len(X_train)*validate_split)
        X_validate = X_train[val_len:]
        Y_validata = Y_train[val_len:]
        
        X_train = X_train[:val_len]
        Y_train = Y_train[:val_len]

    return np.array(X_train), np.array(Y_train), np.array(X_validate), np.array(Y_validata)

def TestDataGenerator(data_path, size, classes, channel):
    
    data_path = join(data_path, 'test_data')
    
    images_dir = join(data_path, 'images')
    labels_dir = join(data_path, 'labels')
    
    data_files = os.listdir(images_dir)
    random.shuffle(data_files)
    
    X_train = [] # grayscale
    Y_train = [] # one-hot vector
    
    landmark_detector = loadLBFmodel()
    
    for data_file in data_files:
        
        img_path = os.path.join(images_dir, data_file)
        label_path = os.path.join(labels_dir, data_file[:-4]+'.txt')
        
        img = cv2.imread(img_path)
        f = open(label_path, "r")
        label = f.read()
        label = label.strip().split()
        
        for i in range(int(len(label)/5)):
            # crop face area
            face = img[int(label[i*5+1]):int(label[i*5+4]), int(label[i*5+2]):int(label[i*5+3]),:]
            face_label = int(label[i*5+0])
            face_out, label_out = face_processing_baseline(face, size, face_label, classes, channel, landmark_detector)
            
            # save X and Y
            X_train.append(face_out)
            Y_train.append(label_out)

    return np.array(X_train), np.array(Y_train)