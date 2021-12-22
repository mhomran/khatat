from preprocessing import preprocess
from featureExtraction import slidingWindowFeatures
from model import Model

from concurrent.futures import ProcessPoolExecutor

import pickle

def process_img(file_path):
    preprocessed, grayimg = preprocess(file_path)
    output = slidingWindowFeatures(preprocessed, grayimg)
    return output

def train_model(model, train):
    for fnt in range(Model.FONTS_NO):
        X = []
        file_paths = []
        for file_name in train[fnt]:
            file_path = f"DBs/ACdata_base/{fnt+1}/{file_name}"
            file_paths.append(file_path)

        with ProcessPoolExecutor() as executor:
            generator = executor.map(process_img, file_paths)

            for fv in generator:
                X.extend(fv)

        model.fit(X, fnt)
        # np.savetxt(f"data{fnt+1}.csv",X,delimiter=',')

    # save
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)