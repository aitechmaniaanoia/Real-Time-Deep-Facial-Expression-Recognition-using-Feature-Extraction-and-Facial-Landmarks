# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:15:13 2020

@author: Zoe
"""

import os
import numpy as np
import cv2

import config
from model import *
from dataset import *
import utils

from keras.models import load_model


model_path = 'output/experiment_2/.mdl_wts.hdf5'

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
fontColor = (255,0,0)
lineType = 2

# model = EmotionVGGNet.build(width = config.SIZE, height = config.SIZE, depth = config.NUM_CHANNEL+1,
#     classes = config.NUM_CLASSES)
model = EmotionVGGNet.build(width = config.SIZE, height = config.SIZE, depth = config.NUM_CHANNEL,
    classes = config.NUM_CLASSES)

#model = load_model(model_path,custom_objects={'f1_m':utils.f1_m, 'precision_m': utils.precision_m, 'recall_m': utils.recall_m})
model = load_model(model_path)

landmark_detector = loadLBFmodel()

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
fontColor = (255,0,0)
lineType = 2

images_dir = 'dataset/test_data/images'
data_files = os.listdir(images_dir)

saved_path = 'dataset/test_data/output_samples/'

for data_file in data_files:
    img_path = os.path.join(images_dir, data_file)
        
    img = cv2.imread(img_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        
        face = gray[y:y+h, x:x+w]
        
        # pre-processing
        face = cv2.resize(face, (config.SIZE, config.SIZE))
        
        norm_face = face/255
        norm_face = norm_face.reshape((1, config.SIZE, config.SIZE, config.NUM_CHANNEL))
        
        # # extract landmarks
        # bbox = np.array([[0,0,face.shape[0],face.shape[1]]])

        # # Detect landmarks on "gray"
        # _, landmarks = landmark_detector.fit(face, bbox)
        
        # landmarks = landmarks[0][0]
        
        # # remove jaw index 0 to 17 
        # landmarks = landmarks[18:,:]
        
        # idx1 = np.where(landmarks[:,0] > face.shape[1])[0]
        # idx2 = np.where(landmarks[:,1] > face.shape[0])[0]
        # idx = np.concatenate((idx1, idx2), axis=0)
        # if len(idx) > 0:
        #     # remove landmark
        #     landmarks = np.delete(landmarks, idx, axis=0)
        
        # landmarks_layer = np.zeros(face.shape[:2])
        # landmarks_layer[landmarks[:,1].astype(int), landmarks[:,0].astype(int)] = 1
        
        # face_out = np.zeros((config.SIZE, config.SIZE, config.NUM_CHANNEL+1))
        # face_out[:,:,:-1] = norm_face
        # face_out[:,:,-1] = landmarks_layer
        
        # face_out = face_out.reshape((1, config.SIZE, config.SIZE, config.NUM_CHANNEL+1))
        
        # predict emotion label
        # pred = model.predict(face_out)
        pred = model.predict(norm_face)
        label = utils.label_num2txt(pred)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
        
        # add label     
        bottomLeftCornerOfText = (x,y-10)
        
        cv2.putText(img,label, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
    # same img  
    
    cv2.imwrite(saved_path + data_file, img)