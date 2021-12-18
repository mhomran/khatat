
import cv2
from preprocessing import binraization,gammaCorrection
from featureExtraction import getBaseline,slidingWindowFeatures,getHOG
import os
from model import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
file_path = f"ACdata_base/3/0390.jpg"
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
grayimg = np.copy(img)
# img =gammaCorrection(img,0.8)


histg = cv2.calcHist([img],[0],None,[256],[0,256])



background=np.argmax(histg)
# print(background)
if background<100:
    img =255-img

# img = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2), interpolation = cv2.INTER_AREA)

img = binraization(img,t=30)
# (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#     cv2.THRESH_BINARY,11,20)


cv2.imshow("img after binrization",img.astype(np.uint8))

cv2.waitKey(0)
# print(file_path)
output = slidingWindowFeatures(img,grayimg)
# output = output.mean(axis=0)
# histogram of gradients
# print(output.shape)
# newFeatures = np.concatenate((output,HOGs))
