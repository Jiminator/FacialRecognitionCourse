#This was a early test to see it the detect method works, which is the method in charge of finding a human face when given a image. This obviously is the simplest of the 4 subsets of my code for this project. There is a small number of lines of code, 0 classes, very shallow call depth, and no loops. The run time is actually quite long, because the program takes a noticeable amount of time displaying the image with a rectangle surrounding the face. 

import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from FaceAPI_dlib import *

path_align_model = 'shape_predictor_5_face_landmarks.dat'
path_verify_model = 'dlib_face_recognition_resnet_model_v1.dat'

img = cv2.imread('data/2.png')

faceAPI = FaceAPI()
faceAPI.loadAlignModel(path_align_model)
faceAPI.loadVerifyModel(path_verify_model)

rects = faceAPI.detect(img)
print(rects)
print(rects[0].p1)
cv2.rectangle(img,(rects[0].p1.x,rects[0].p1.y),(rects[0].p2.x,rects[0].p2.y),(0,0,200),2)
plt.figure()
plt.imshow(img[:,:,::-1])
plt.show()
#lms = faceAPI.align(img,rects[0])
#print(lms)
#faceAPI.extractFeature(img,rects[0])

