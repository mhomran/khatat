import numpy as np
from numpy.core.records import array 
import cv2

windowWidth = 8 
numberOfCells = 3
def getBaseline(img):
    
    hProjection = np.sum(1-img/255,axis=1)
    h =len(hProjection)
    hProjection = hProjection[h//10:]
    mean = np.mean(hProjection)
    LB = np.argmax(hProjection)+h//10
    UB = np.argmax(hProjection>=mean)+h//10
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
    # h = 4 
    x1 = img.shape[0]-1
    x2 = x1-w
    pervg = getCenterOfGravity(img[max(x2,0):x1,:])+np.array([w,0])
    # loop through windows
    while x1 >0:
        window = img[max(x2,0):x1,:]
        graywindow = grayimg[max(x2,0):x1,:]
        x1 -= 1
        x2 -= 1
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
        # leftup concavity 
        f10 = leftupConcavity(window)/H
        # rightup concavity 
        f11 = rightupConcavity(window)/H
        # rightdown concavity 
        f12 = rightdownConcavity(window)/H
        # leftdown concavity
        f13 = leftdownConcavity(window)/H
        #virtcal concavity
        f14 = verticalConcavity(window)/H

        #horizontal concavity
        f15 = horizontalConcavity(window)/H
        

        # core zone
        # leftup concavity 

        coreZone = np.copy(window[:,UB:LB+3])
        try:
            f16 = leftupConcavity(coreZone)/H
        except:
            print(LB,UB)
        # rightup concavity 
        f17 = rightupConcavity(coreZone)/H
        # rightdown concavity 
        f18 = rightdownConcavity(coreZone)/H
        # leftdown concavity
        f19 = leftdownConcavity(coreZone)/H
        #virtcal concavity
        f20 = verticalConcavity(coreZone)/H

        #horizontal concavity
        f21 = horizontalConcavity(coreZone)/H


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

        
        current_feature_vector= [LB,UB,f1,f2,f3,f4,f5,f6,f7,f8,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21]
        current_feature_vector+=list(f21to28)
        # current_feature_vector+=(getHOG(graywindow))
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

def leftupConcavity(img):
    
    k1 = np.array([
        [0,1,0],
        [1,0,0],
        [0,0,0]
    ],np.uint8)
    k2 = np.array([
        [0,0,1],
        [0,1,1],
        [1,1,0]
    ],np.uint8)
    out1 = cv2.erode(img, k1, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 0 )
    out2 = cv2.erode(1-img, k2, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 1 )
    return np.sum(out1*out2)

def rightupConcavity(img):
    k1 = np.array([
        [0,1,0],
        [0,0,1],
        [0,0,0]
    ],np.uint8)
    k2 = np.array([
        [1,0,0],
        [1,1,0],
        [0,1,1]
    ],np.uint8)
    out1 = cv2.erode(img, k1, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 0 )
    out2 = cv2.erode(1-img, k2, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 1 )
    return np.sum(out1*out2)

def rightdownConcavity(img):
    k1 = np.array([
        [0,0,0],
        [0,0,1],
        [0,1,0]
    ],np.uint8)
    k2 = np.array([
        [0,1,1],
        [1,1,0],
        [1,0,0]
    ],np.uint8)
    out1 = cv2.erode(img, k1, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 0 )
    out2 = cv2.erode(1-img, k2, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 1 )
    return np.sum(out1*out2)   

def leftdownConcavity(img):
    k1 = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0]
    ],np.uint8)
    k2 = np.array([
        [1,1,0],
        [0,1,1],
        [0,0,1]
    ],np.uint8)
    out1 = cv2.erode(img, k1, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 0 )
    out2 = cv2.erode(1-img, k2, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 1 )
    return np.sum(out1*out2)   

def verticalConcavity(img):
    k1 = np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0]
    ],np.uint8)
    k2 = np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ],np.uint8)
    out1 = cv2.erode(img, k1, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 0 )
    out2 = cv2.erode(1-img, k2, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 1 )
    return np.sum(out1*out2)  


def horizontalConcavity(img):
    k1 = np.array([
        [0,0,0],
        [0,0,0],
        [1,1,1]
    ],np.uint8)
    k2 = np.array([
        [0,0,0],
        [1,1,1],
        [0,0,0]
    ],np.uint8)
    out1 = cv2.erode(img, k1, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 0 )
    out2 = cv2.erode(1-img, k2, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue = 1 )
    return np.sum(out1*out2)  