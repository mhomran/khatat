
import os, pickle, sys
from model import Model
from trainModule import train_model
from testModule import test_model, classify
from sklearn.model_selection import train_test_split

def main(pretrained = None, want_classify = None, data_folder_path = "data", output_folder_path = "out"):

    model = Model()
    train = []
    test = []

    while pretrained != 'y' and pretrained != 'n':
        pretrained = input("Do you want a pretrained model ? (y/n)\n")

    while want_classify != 'y' and want_classify != 'n':
        want_classify = input("Do you to want to classify data ? (y/n)\n")

    if pretrained == 'y':
        # load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        if want_classify == 'n': # all test and train data is set as test data
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
        test = os.listdir(data_folder_path)
        classify(model, test, output_folder_path)
    else: test_model(model, test)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        pretrained = 'y'
        want_classify = 'y'
        data_folder_path = sys.argv[1]
        output_folder_path = sys.argv[2]
        
        main(pretrained, want_classify, data_folder_path, output_folder_path)
    else:
        main()
