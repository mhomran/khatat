from preprocessing import preprocess
from featureExtraction import slidingWindowFeatures
import os, time, pickle
from model import Model
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def process_img(file_path):
    preprocessed, grayimg = preprocess(file_path)
    output = slidingWindowFeatures(preprocessed, grayimg)
    return output

def predict_img(file_path, model):
    start_time = time.perf_counter()

    output = process_img(file_path)
    label = model.predict(output)

    finish_time = time.perf_counter()

    return label, (finish_time-start_time)

def main():
    labels_to_strings = {
        1: "diwani",
        2: "naskh",
        3: "parsi",
        4: "rekaa",
        5: "thuluth",
        6: "maghribi",
        7: "kufi",
        8: "mohakek",
        9: "Squar-kufic"
    }
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


    # test
    f_result = open("results.txt", "w")
    f_time = open("time.txt", "w")
    
    start = time.perf_counter()

    total_cclassified = 0
    total_test_no = 0
    for fnt in range(Model.FONTS_NO):
        correctly_classified = 0

        labels = []
        with ProcessPoolExecutor() as executor:            
            for file_name in test[fnt]:
                file_path = f"DBs/ACdata_base/{fnt+1}/{file_name}"
                labels.append(executor.submit(predict_img, file_path, model))
        
        i = 0
        for label in labels:
            i += 1
            
            label, img_time = label.result()

            if label == fnt: 
                correctly_classified += 1
                total_cclassified += 1
            else:
                print(f"{file_path} GT: {labels_to_strings[fnt+1]}, predicted: {labels_to_strings[label+1]}")
                # cv2.imshow("false prediction",preprocessed)
                # cv2.waitKey(0)

            f_result.write(str(label))
            f_time.write(str(img_time/cpu_count()))
            if i != len(test[fnt]): 
                f_result.write('\n')
                f_time.write('\n')

        accuracy = correctly_classified / len(test[fnt])
        total_test_no += len(test[fnt])
        print(f"font {fnt+1} accuracy is {accuracy}")
    print(f"total accuracy is {total_cclassified/total_test_no}")
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} seconds(s)")
    print(f"Image inference time is: {(finish - start)/total_test_no} seconds")

    f_result.close()
    f_time.close()

if __name__ == '__main__':
    main()
