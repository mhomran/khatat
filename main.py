import cv2
from preproccessing import binraization
from featuresExtraction import getBaseline,slidingWindowFeatures
for i in range(1,9):

    file = "ACdata_base/1/001{}.jpg".format(i)
    print(file)
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img[0,0]<150:
        img =255-img
    cv2.imwrite("testing/{}orignial.jpg".format(i),img)
    img = binraization(img,t=30)
    (lowerBaseline,upperBaseline) = getBaseline(img)

    cv2.line(img,(0,lowerBaseline),(img.shape[1],lowerBaseline),(0,0,0),2)
    cv2.line(img,(0,upperBaseline),(img.shape[1],upperBaseline),(0,0,0),2)
    output = slidingWindowFeatures(img)
    cv2.imwrite("testing/{}.jpg".format(i),img)