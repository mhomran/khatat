import cv2
from preprocessing import binraization
from featureExtraction import getBaseline,slidingWindowFeatures,getHOG
from skimage.filters import threshold_otsu, threshold_local
import os
from model import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
# seperate test and train data
train = []
test = []
for fnt in range(Model.FONTS_NO):
    file_names = os.listdir(f"ACdata_base/{fnt+1}")
    both = train_test_split(file_names, test_size=.2)
    train.append(both[0])
    test.append(both[1])

# train
model = Model()
for fnt in range(Model.FONTS_NO):
    X = []
    for file_name in train[fnt]:
        file_path = f"ACdata_base/{fnt+1}/{file_name}"
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        grayimg = np.copy(img)
        ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        histg = cv2.calcHist([img],[0],None,[256],[0,256]) 
        background = np.argmax(histg)
     

        if background<50:
            img =255-img
        output = slidingWindowFeatures(img,grayimg)
        
        X.extend(output)

    


    # X =preprocessing.normalize(X)
    model.fit(X, fnt)

# test
accuracies = []
myacc =0
total =0
for fnt in range(Model.FONTS_NO):
    correctly_classified = 0
    for file_name in test[fnt]:
        file_path = f"ACdata_base/{fnt+1}/{file_name}"
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        grayimg = np.copy(img)
        
        histg = cv2.calcHist([img],[0],None,[256],[0,256]) 
        background = np.argmax(histg)
        
        if background<50:
            img =255-img

        
        ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        output = slidingWindowFeatures(img,grayimg)

        label = model.predict(output)
        if label == fnt: correctly_classified += 1

    accuracy = correctly_classified / len(test[fnt])
    myacc+=correctly_classified
    total+=len(test[fnt])
    accuracies.append(accuracy)
    print(f"font {fnt+1} accuracy is {accuracy}")
print(f"total accurcy {myacc/total}")
print(f"total accuracy is {np.mean(accuracies)}")
