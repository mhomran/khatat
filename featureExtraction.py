import numpy as np
from numpy.core.records import array 
import cv2

windowWidth = 8 
numberOfCells = 20
def getBaseline(img):
    hProjection = np.sum(1-img/255,axis=1)
    mean = np.mean(hProjection)
    LB = np.argmax(hProjection)
    UB = np.argmax(hProjection>=mean)
    return LB,UB


def getCenterOfGravity(window):
    indecies = np.argwhere(window == 1)
    if(len(indecies)==0):
        return np.zeros(2)
    centerOfGravity = np.mean(indecies,axis =0)
    return centerOfGravity


def slidingWindowFeatures(img,grayimg):
    features = []
    # calc base line
    LB , UB = getBaseline(img)
    # background =0, forground =1 
    img = 1-img/255
    w = windowWidth 
    H = img.shape[1]
    n = numberOfCells
    h = H//n
    x1 = img.shape[0]-1
    x2 = x1-w
    pervg = getCenterOfGravity(img[max(x2,0):x1,:])+np.array([w,0])
    # loop through windows
    while x1 >0:
        window = img[max(x2,0):x1,:]
        graywindow = grayimg[max(x2,0):x1,:]
        x1 = x2
        x2 -= w 
        f1 = 0
        f2 = 0
        y1 = H
        y2 = H-h
        bi1 =0
        # r(j): the number of foreground pixels in the jth row of a frame.
        r = np.sum(window,axis=0)
        # g center of gravity 
        g = getCenterOfGravity(window)
        # f3 = g(t)-g(t-1)
        f3,f4 = (g - pervg)
        # f4 = (g-L)/H
        f5,f6 = ((g-LB)/H)
        # f5 = sum(r(j)) from L+1 to h / H*W 
        f7 = np.sum(r[LB+1:H])/H*w
        # f6 = sum(r(j)) from 1 to L-1 / H*W 
        f8 = np.sum(r[:LB])/H*w
        # loop through cells


        f9 =(g[1]/H)*3

        cells = []
        while y1>0:
            # slicing the cell
            cells.append(window[:,max(y2,0):y1])
            y1 = y2
            y2-= h
        # f1 = sum ni/n    
        f1 = 0
        # f2 = sum (|bi - bi-1|)
        f2 = 0 
        for i in range(1,len(cells)):
            # n(i): the number of foreground pixels in cell i
            ni = np.sum(cells[i])
            # b(i): the density level of cell i: b(i) = 0 if n(i) = 0 else b(i) = 1
            bi1 = bool(ni)
            bi0 = bool(np.sum(cells[i-1]))
            f1 += ni 
            f2 += abs(bi0 - bi1)    
        f1 /= (H*w)
        f21to28 = np.sum(window,axis =1)
        
        if(len(f21to28)<8):
            f21to28 = np.concatenate((f21to28,np.zeros(8-len(f21to28))))

        
        current_feature_vector= [LB,UB,f1,f2,f3,f4,f5,f6,f7,f8]
        current_feature_vector+=list(f21to28)
        current_feature_vector+=(getHOG(graywindow))
        features.append(current_feature_vector)
    return np.array(features)
    
def getHOG(img):
    hist = [0]*8
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle /= 45
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            hist[np.uint16(angle[i,j])] += 1
    return hist