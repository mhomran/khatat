import imutils
import numpy as np
import cv2
from scipy.signal import find_peaks

WINDOW_WIDTH = 15
CELLS_NO = 3
WINDOW_SHIFT = WINDOW_WIDTH//2
EPS = 1e-10
def getBaseline(img):
    hProjection = np.sum(img,axis=1)
    mean = np.mean(hProjection)
    LB = np.argmax(hProjection)
    UB = np.argmax(hProjection>=mean)
    return LB, UB

def getCenterOfGravity(window):
    indecies = np.argwhere(window == 1)
    if(len(indecies)==0):
        return np.zeros(2)
    centerOfGravity = np.mean(indecies, axis=0)
    return centerOfGravity


def slidingWindowFeatures(img,grayimg):
    features = []
    # background = 0, forground =1 
    img = 1-img//255
    # calc base line
    LB, UB = getBaseline(img)
    w = WINDOW_WIDTH 
    H = img.shape[0]
    x1 = img.shape[1]-1 # last idx
    x2 = x1 - w # last idx - window size
    prevg = getCenterOfGravity(img[:, max(x2,0):x1])
    # loop through windows
    while x1 > 0:
        window = img[:, max(x2,0):x1]
        if window.shape[1] != w: break
        x1 -= WINDOW_SHIFT
        x2 -= WINDOW_SHIFT
        # r(j): the number of foreground pixels in the jth row of a frame.
        r = np.sum(window, axis=1)
        # g center of gravity 
        g = getCenterOfGravity(window)
        f1 = np.sum(window)/(H*w)
        # f3 = g(t)-g(t-1)
        f3,f4 = (g - prevg)
        prevg = g
        # f4 = (g-L)/H
        f5,f6 = ((g-LB)/H)
        # f5 = sum(r(j)) from L+1 to h / H*W 
        f7 = np.sum(r[LB+1:H])/(H*w)
        # f6 = sum(r(j)) from 1 to L-1 / H*W 
        f8 = np.sum(r[:LB-1])/(H*w)
        f9 =(g[1]/H)*3

        # concavity configurations
        f10tof15 = get_concavity_features(window, H)
        # core zone
        core_zone = window[UB:LB+3, :]
        f16tof21 = get_concavity_features(core_zone, H)

        f21to28 = np.sum(window,axis =1)
        
        if(len(f21to28)<w):
            f21to28 = np.concatenate((f21to28,np.zeros(w-len(f21to28))))


        black_contours, _ = cv2.findContours(window, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        white_contours, _ = cv2.findContours(1-window, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        N1 = len(black_contours)
        N2 = len(white_contours)
        N1_N2 = N1/(N2+EPS)

        # compactness = 0
        # black_contours = list(black_contours)
        # if len(black_contours) > 0:
        #     for cnt in black_contours:
        #         perimeter = cv2.arcLength(cnt, True)
        #         area = cv2.contourArea(cnt)
        #         compactness += perimeter * (np.pi * (area ** 2))

        hProjection = np.sum(window, axis=1)
        h_extremas = len(find_peaks(hProjection, height=0)[0])
        vProjection = np.sum(np.transpose(window), axis=1)
        v_extremas = len(find_peaks(vProjection, height=0)[0])

        # WARNING: window is normalized
        # window = imutils.resize(window, height=20)
        # hProjection = list(np.sum(window, axis=1))
        # vProjection = list(np.sum(np.transpose(window), axis=1))

        current_feature_vector= [f1,f3,f4,f5,f6,f7,f8, 
                                *f10tof15, 
                                *f16tof21,
                                N1, N2, N1_N2,
                                h_extremas, v_extremas]
                                # *hProjection, *vProjection,
                                # *f21to28]
        # current_feature_vector+=(getHOG(graywindow))
        # current_feature_vector+=list(f21to28)
        features.append(current_feature_vector)
    return np.array(features)
    
def leftup_concavity(img):
    
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

def rightup_concavity(img):
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

def rightdown_concavity(img):
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

def leftdown_concavity(img):
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

def vertical_concavity(img):
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


def horizontal_concavity(img):
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

def get_concavity_features(window, H):
    # leftup concavity 
    flu = leftup_concavity(window)/H
    # rightup concavity 
    fru = rightup_concavity(window)/H
    # rightdown concavity 
    frd = rightdown_concavity(window)/H
    # leftdown concavity
    fld = leftdown_concavity(window)/H
    #virtcal concavity
    fv = vertical_concavity(window)/H
    #horizontal concavity
    fh = horizontal_concavity(window)/H

    return [flu, fru, frd, fld, fv, fh]

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