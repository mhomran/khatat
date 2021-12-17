import cv2
from preprocessing import binraization
from featureExtraction import getBaseline,slidingWindowFeatures
import os
from model import Model
from sklearn.model_selection import train_test_split
import numpy as np

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

        if img[0,0]<150:
            img =255-img

        img = binraization(img,t=30)

        output = slidingWindowFeatures(img)

        X.extend(output)

    model.fit(X, fnt)

# test

# train
accuracies = []
for fnt in range(Model.FONTS_NO):
    correctly_classified = 0
    for file_name in test[fnt]:
        file_path = f"ACdata_base/{fnt+1}/{file_name}"
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img[0,0]<150:
            img =255-img

        img = binraization(img,t=30)

        output = slidingWindowFeatures(img)

        label = model.predict(output)
        if label == fnt: correctly_classified += 1

    accuracy = correctly_classified / len(test[fnt])
    accuracies.append(accuracy)
    print(f"font {fnt} accuracy is {accuracy}")

print(f"total accuracy is {np.mean(accuracies)}")
