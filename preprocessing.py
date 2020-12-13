# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:56:46 2020

@author: Zoe
"""

import numpy as np
import cv2
import face_alignment

import os
import urllib.request as urlreq

# ## face alignment
#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face = cv2.imread('face1.jpg')
gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

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
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)


# each face
bbox = np.array([[0,0,gray.shape[0],gray.shape[1]]])

# Detect landmarks on "gray"
_, landmarks = landmark_detector.fit(gray, bbox)

landmarks = landmarks[0][0]

# remove jaw index 0 to 17 
landmarks = landmarks[18:,:]

# remove landmarks outside image
idx1 = np.where(landmarks[:,0] > face.shape[1])[0]
idx2 = np.where(landmarks[:,1] > face.shape[0])[0]
idx = np.concatenate((idx1, idx2), axis=0)
if len(idx) > 0:
    # remove landmark
    landmarks = np.delete(landmarks, idx, axis=0)

landmarks_layer = np.zeros(face.shape[:2])
landmarks_layer[landmarks[:,1].astype(int), landmarks[:,0].astype(int)] = 1


# FACIAL_LANDMARKS_IDXS = OrderedDict([
# 	("mouth", (48, 68)),
# 	("right_eyebrow", (17, 22)),
# 	("left_eyebrow", (22, 27)),
# 	("right_eye", (36, 42)),
# 	("left_eye", (42, 48)),
# 	("nose", (27, 35)),
# 	("jaw", (0, 17))
# ])



# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = faceCascade.detectMultiScale(
#     gray,     
#     scaleFactor=1.2,
#     minNeighbors=5,     
#     minSize=(20, 20)
# )

# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]  
# cv2.imshow('face',img)


# # cap = cv2.VideoCapture(0)
# # cap.set(3,640) # set Width
# # cap.set(4,480) # set Height
# # while True:
# #     ret, img = cap.read()
# #     img = cv2.flip(img, -1)
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     faces = faceCascade.detectMultiScale(
# #         gray,     
# #         scaleFactor=1.2,
# #         minNeighbors=5,     
# #         minSize=(20, 20)
# #     )
# #     for (x,y,w,h) in faces:
# #         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# #         roi_gray = gray[y:y+h, x:x+w]
# #         roi_color = img[y:y+h, x:x+w]  
# #     cv2.imshow('video',img)
# #     k = cv2.waitKey(30) & 0xff
# #     if k == 27: # press 'ESC' to quit
# #         break
# # cap.release()
# # cv2.destroyAllWindows()


# # ## illumination normalization
# hh, ww = img.shape[:2]
# max = max(hh, ww)

# # illumination normalize
# ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# # separate channels
# y, cr, cb = cv2.split(ycrcb)

# # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
# # account for size of input vs 300
# sigma = int(5 * max / 300)
# gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

# # subtract background from Y channel
# y = (y - gaussian + 100)

# # merge channels back
# ycrcb = cv2.merge([y, cr, cb])

# #convert to BGR
# output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# # show results
# cv2.imshow("output", output)


# ## split label into txt file
# f = open('label.lst', 'r')
# lines = f.readlines()
# for ln in lines:
#     line = ln.strip().split()
#     label_name = line[0][:-4] + '.txt'
    
#     # save label information
#     f2_name = 'dataset/labels/' + label_name
#     f2 = open(f2_name, 'a')
#     f2.write(line[-1]+' '+line[2]+' '+line[3]+' '+line[4]+' '+line[5]+'\n')
#     f2.close()

#     ## save image with label
#     img = cv2.imread('dataset/images_all/'+line[0])
#     cv2.imwrite('dataset/images/'+line[0], img)
# f.close()

## split trainingand test files
# import os
# from os.path import join
# import config
# import random 
# import shutil

# data_path = config.BASE_PATH

# images_dir = join(data_path, 'images')
# labels_dir = join(data_path, 'labels')

# dest_image_dir = join(data_path, 'test_data/images')
# dest_label_dir = join(data_path, 'test_data/labels')


# data_files = os.listdir(images_dir)
# random.shuffle(data_files)

# # random pick 10,000 images for test
# test_idx = random.sample(range(0, len(data_files)), 10000)
# for idx in test_idx:
#     data_file = data_files[idx]
    
#     img_path = os.path.join(images_dir, data_file)
#     label_path = os.path.join(labels_dir, data_file[:-4]+'.txt')
    
#     dest_img_path = os.path.join(dest_image_dir, data_file)
#     dest_label_path = os.path.join(dest_label_dir, data_file[:-4]+'.txt')
    
#     shutil.move(img_path, dest_img_path)
#     shutil.move(label_path, dest_label_path)










