# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:32:28 2020

@author: Zoe
"""

import numpy as np
import cv2
import argparse
from keras.models import load_model
# from tensorflow.keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam

import utils
import config
from model import *
from dataset import *

exp = 1

if exp == 1:
    model_path = 'output/experiment_1/.mdl_wts.hdf5'
elif exp == 2:
    model_path = 'output/experiment_2/.mdl_wts.hdf5'
elif exp == 3:
    model_path = 'output/experiment_3/.mdl_wts.hdf5'
elif exp == 4:
    model_path = 'output/experiment_4/.mdl_wts.hdf5'

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

# compile model
if exp <= 2:
    model = EmotionVGGNet.build(width = config.SIZE, height = config.SIZE, depth = config.NUM_CHANNEL,
        classes = config.NUM_CLASSES)
else:
    model = EmotionVGGNet.build(width = config.SIZE, height = config.SIZE, depth = config.NUM_CHANNEL+1,
        classes = config.NUM_CLASSES)
    
# opt = Adam(lr = 1e-3)
# model.compile(loss = "categorical_crossentropy", optimizer = opt,
#     metrics=['acc', utils.f1_m, utils.precision_m, utils.recall_m])

model = load_model()

landmark_detector = loadLBFmodel()

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
fontColor = (255,0,0)
lineType = 2

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # face alignment
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    
    if faces:
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]/255
            
            # pre-processing
            face = cv2.resize(face, (config.SIZE, config.SIZE))
            
            norm_face = face/255
            face_out = norm_face.reshape((config.SIZE, config.SIZE, config.NUM_CHANNEL))

            if exp > 2:
                # extract landmarks
                bbox = np.array([[0,0,face.shape[0],face.shape[1]]])
    
                # Detect landmarks on "gray"
                _, landmarks = landmark_detector.fit(face, bbox)
                
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
                
                face_out = np.zeros((config.SIZE, config.SIZE, config.NUM_CHANNEL+1))
                face_out[:,:,:-1] = norm_face
                face_out[:,:,-1] = landmarks_layer
            
            else:
                face_out = norm_face
                
            # predict emotion label
            pred = model.predict(face_out)
            
            label = utils.label_num2txt(pred)
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # show label in image
            bottomLeftCornerOfText = (x,y-10)
        
            cv2.putText(img,label, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
        
    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()