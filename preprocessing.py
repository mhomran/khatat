import numpy as np 

def binraization(img,n=8,t=15):

    outputimg = np.zeros(img.shape)
    intimg = np.zeros(img.shape)
    h = img.shape[1]
    w = img.shape[0]
    s= min(w,h)//n
    count = s**2
    img = np.pad(img,s,"constant")
    intimg = np.cumsum(img ,axis =1)
    intimg = np.cumsum(intimg ,axis =0)
    a = np.roll(intimg,-s//2,axis =0)
    a = np.roll(a,-s//2,axis =1)
    a[:,-s//2:]=a[-s//2-1,-s//2-1]
    a[-s//2:,:]=a[-s//2-1,-s//2-1]
    b = np.roll(intimg,s//2+1,axis =0)
    b = np.roll(b,-s//2,axis =1)
    b[0:s//2+1,:]=0
    b[:,-s//2:]=0
    
    c = np.roll(intimg,s//2+1,axis =1)
    c = np.roll(c,-s//2,axis =0)
    c[:,0:s//2+1]=0
    c[-s//2:,:]=0
    
    d = np.roll(intimg,s//2+1,axis =0)
    d = np.roll(d,s//2+1,axis =1)
    d[0:s//2+1,:]=0
    d[:,0:s//2+1]=0

    sum = (a-b-c+d)*(100-t)/100
    outputimg = (img>sum/count)*255
    return outputimg[s:-s,s:-s]
import cv2, imutils

def preprocess(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    grayimg = np.copy(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img = imutils.resize(img, height=45)
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