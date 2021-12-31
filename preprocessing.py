import numpy as np 
import cv2

def preprocess(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    grayimg = np.copy(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    counts = np.bincount(img.flatten())
    background = np.argmax(counts)
    if background < 50:
        img = 255 - img
    img = remove_trailing_lines(img) 
    
    return img,grayimg


def remove_trailing_lines(img):
    img = 255-img
    hProjection = np.sum(img,axis=1)
    minY = np.argmax(hProjection>0)
    hProjection = hProjection[::-1]
    maxY = img.shape[0] - np.argmax(hProjection>0)
    
    vProjection = np.sum(img.T,axis=1)
    minX = np.argmax(vProjection>0)
    vProjection = vProjection[::-1]
    maxX = img.shape[1] - np.argmax(vProjection>0)
    img = 255-img
    return img[minY:maxY,minX:maxX]