from preprocessing import preprocess
from featureExtraction import slidingWindowFeatures
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

        preprocessed = preprocess(file_path)
        output = slidingWindowFeatures(preprocessed)
        
        X.extend(output)

    model.fit(X, fnt)
    np.savetxt(f"data{fnt+1}.csv",X,delimiter=',')
# test
accuracies = []
total =0
for fnt in range(Model.FONTS_NO):
    correctly_classified = 0
    for file_name in test[fnt]:
        file_path = f"ACdata_base/{fnt+1}/{file_name}"
        preprocessed = preprocess(file_path)
        output = slidingWindowFeatures(preprocessed)

        label = model.predict(output)
        if label == fnt: correctly_classified += 1
        # else:
            # print(label)
            # print(file_name)
            # cv2.imshow("false prediction",img)
            # cv2.waitKey(0)
    accuracy = correctly_classified / len(test[fnt])
    total+=len(test[fnt])
    accuracies.append(accuracy)
    print(f"font {fnt+1} accuracy is {accuracy}")
print(f"total accuracy is {np.mean(accuracies)}")
