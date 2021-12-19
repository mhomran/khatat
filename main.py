import cv2
from preprocessing import preprocess
from featureExtraction import slidingWindowFeatures
import os
from model import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

model = Model()
train = []
test = []

pretrained = None
while pretrained != 'y' and pretrained != 'n':
    pretrained = input("Do you want it pretrained ? (y/n)\n")

if pretrained == 'y':
    # load model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    for fnt in range(Model.FONTS_NO):
        file_names = os.listdir(f"ACdata_base/{fnt+1}")
        test.append(file_names)
else:
    # seperate test and train data
    for fnt in range(Model.FONTS_NO):
        file_names = os.listdir(f"ACdata_base/{fnt+1}")
        both = train_test_split(file_names, test_size=.2)
        train.append(both[0])
        test.append(both[1])

    # train
    for fnt in range(Model.FONTS_NO):
        X = []
        for file_name in train[fnt]:
            file_path = f"ACdata_base/{fnt+1}/{file_name}"

            preprocessed,grayimg = preprocess(file_path)
            # print(grayimg.shape)
            output = slidingWindowFeatures(preprocessed,grayimg)
            
            X.extend(output)

        model.fit(X, fnt)
        # np.savetxt(f"data{fnt+1}.csv",X,delimiter=',')

    # save
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)


# test
accuracies = []
total =0
for fnt in range(Model.FONTS_NO):
    correctly_classified = 0
    for file_name in test[fnt]:
        file_path = f"ACdata_base/{fnt+1}/{file_name}"
        preprocessed,grayimg = preprocess(file_path)
        output = slidingWindowFeatures(preprocessed,grayimg)

        label = model.predict(output)
        if label == fnt: correctly_classified += 1
        else:
            print(f"GT: {fnt+1}, predicted: {label+1}")
            print(file_name)
            cv2.imshow("false prediction",preprocessed)
            cv2.waitKey(0)
    accuracy = correctly_classified / len(test[fnt])
    total+=len(test[fnt])
    accuracies.append(accuracy)
    print(f"font {fnt+1} accuracy is {accuracy}")
print(f"total accuracy is {np.mean(accuracies)}")
