# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:56:46 2020

@author: Zoe
"""

import numpy as np
import cv2


# ## face alignment
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('faces.jpg')
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


# ## illumination normalization
hh, ww = img.shape[:2]
max = max(hh, ww)

# illumination normalize
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# separate channels
y, cr, cb = cv2.split(ycrcb)

# get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
# account for size of input vs 300
sigma = int(5 * max / 300)
gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

# subtract background from Y channel
y = (y - gaussian + 100)

# merge channels back
ycrcb = cv2.merge([y, cr, cb])

#convert to BGR
output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# show results
cv2.imshow("output", output)


## split label into txt file
f = open('label.lst', 'r')
lines = f.readlines()
for ln in lines:
    line = ln.strip().split()
    label_name = line[0][:-4] + '.txt'
    
    # save label information
    f2_name = 'dataset/labels/' + label_name
    f2 = open(f2_name, 'a')
    f2.write(line[-1]+' '+line[2]+' '+line[3]+' '+line[4]+' '+line[5]+'\n')
    f2.close()

    ## save image with label
    img = cv2.imread('dataset/images_all/'+line[0])
    cv2.imwrite('dataset/images/'+line[0], img)
f.close()









