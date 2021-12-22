
import os, pickle

from model import Model
from trainModule import train_model
from testModule import test_model, classify

from sklearn.model_selection import train_test_split

def main():

    model = Model()
    train = []
    test = []

    pretrained = None
    while pretrained != 'y' and pretrained != 'n':
        pretrained = input("Do you want a pretrained model ? (y/n)\n")

    want_classify = None
    while want_classify != 'y' and want_classify != 'n':
        want_classify = input("Do you to want to classify data ? (y/n)\n")

    if pretrained == 'y':
        # load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        for fnt in range(Model.FONTS_NO):
            file_names = os.listdir(f"DBs/ACdata_base/{fnt+1}")
            test.append(file_names)
    else:
        # seperate test and train data
        for fnt in range(Model.FONTS_NO):
            file_names = os.listdir(f"DBs/ACdata_base/{fnt+1}")
            both = train_test_split(file_names, test_size=.2)
            train.append(both[0])
            test.append(both[1])

        # train
        train_model(model, train)
    
    # test or classify
    if want_classify == 'y': 
        test = os.listdir("data")
        classify(model, test)
    else: test_model(model, test)

if __name__ == '__main__':
    main()
