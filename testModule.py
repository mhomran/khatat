from preprocessing import preprocess
from featureExtraction import slidingWindowFeatures
from model import Model

from time import perf_counter

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count



def predict_img(file_path, model):
    start_time = perf_counter()

    preprocessed, grayimg = preprocess(file_path)
    output = slidingWindowFeatures(preprocessed, grayimg)
    label = model.predict(output)

    finish_time = perf_counter()

    return label, (finish_time-start_time)

def test_model(model, test):
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

    f_result = open("out/results.txt", "w")
    f_time = open("out/times.txt", "w")
    
    start = perf_counter()

    total_cclassified = 0
    total_test_no = 0
    for fnt in range(Model.FONTS_NO):
        correctly_classified = 0

        labels = []
        file_pathes = []
        with ProcessPoolExecutor() as executor:            
            for file_name in test[fnt]:
                file_path = f"DBs/ACdata_base/{fnt+1}/{file_name}"
                file_pathes.append(file_path)
                labels.append(executor.submit(predict_img, file_path, model))
        
        for idx, label in enumerate(labels):
            
            label, img_time = label.result()

            if label == fnt: 
                correctly_classified += 1
                total_cclassified += 1
            else:
                print(f"{file_pathes[idx]} GT: {labels_to_strings[fnt+1]}, predicted: {labels_to_strings[label+1]}")
                # cv2.imshow("false prediction",preprocessed)
                # cv2.waitKey(0)

            f_result.write(str(label))
            f_time.write(str(round(img_time/cpu_count(), 2)))
            if idx != len(labels)-1 or fnt != Model.FONTS_NO-1: 
                f_result.write('\n')
                f_time.write('\n')

        accuracy = correctly_classified / len(test[fnt])
        total_test_no += len(test[fnt])
        print(f"font {fnt+1} accuracy is {accuracy}")
    print(f"total accuracy is {total_cclassified/total_test_no}")
    finish = perf_counter()
    print(f"Finished in {round(finish-start, 2)} seconds(s)")
    print(f"Image inference time is: {(finish - start)/total_test_no} seconds")

    f_result.close()
    f_time.close()
    
def classify(model, test, output_folder_path):
    f_result = open(f"{output_folder_path}/results.txt", "w")
    f_time = open(f"{output_folder_path}/times.txt", "w")
    

    labels = []
    with ProcessPoolExecutor() as executor:            
        for file_name in test:
            file_path = f"data/{file_name}"
            labels.append(executor.submit(predict_img, file_path, model))
    
    i = 0
    for label in labels:
        i += 1
        
        label, img_time = label.result()

        f_result.write(str(label + 1))
        f_time.write(str(round(img_time/cpu_count(), 2)))
        if i != len(test): 
            f_result.write('\n')
            f_time.write('\n')
        
    f_result.close()
    f_time.close()
