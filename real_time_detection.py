# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:32:28 2020

@author: Zoe
"""

import numpy as np
import cv2
import argparse
from keras.models import load_model

import utils

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type = str, help = "path to specific model checkpoint to load")
args = vars(ap.parse_args())


faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

model = model = load_model(args["model"])

while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
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
            # predict emotion label
            pred = model.predict(face)
            
            pred = utils.label_num2txt(pred)
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # TODO: show label in image
            
        
    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()