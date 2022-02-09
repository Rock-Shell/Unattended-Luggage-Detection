import cv2
import numpy as np 
import os
import time
import matplotlib.pyplot as plt

def blur(img):
    #return cv2.medianBlur(img, 5)
    #return cv2.bilateralFilter(img, 9, 75,75)
    return cv2.GaussianBlur(img, (5,5),-1)

def BW(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# path = "D:/Intership workAnemoi assignment/left luggage detection/dataset/S1-T1-C/video\pets2006\S1-T1-C/4"
# # path2 = "trial"
# base = "S1-T1-C.00000"

path = "D:/Intership work/Anemoi assignment/left luggage detection/dataset/S1-T1-C/S1-T1-C/video\pets2006/S1-T1-C/4"
path2 = "D:/Intership work/Anemoi assignment/left luggage detection/trial"
base = "S1-T1-C.00000"

start = time.time()

FF = cv2.imread(path+'/' + base + ".jpeg")
# print(FF)
white = 255*np.ones(FF.shape[:2], np.uint8)
black = np.zeros(FF.shape[:2], np.uint8)
mask2 = 255*np.ones(FF.shape[:2], np.uint8)
moving = np.zeros(FF.shape,np.uint8)
kernel2 = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

background = cv2.imread(path2+"/background/background4.jpeg")
background2 = cv2.absdiff(FF, background)
background2 = np.float32(background2)

j=-1
frame2 = FF.copy()
for i in os.listdir(path):
    j+=1
    now = time.time()
    frame = cv2.imread(path + "/" + i)

    mask = cv2.absdiff(BW(frame), BW(background))
    if j%1 == 0:
        moving = cv2.absdiff(frame, frame2)
        frame2 = frame.copy()

        gray = BW(moving)
        gray = blur(blur(blur(gray)))
        gray = 3*gray
        _, thresh = cv2.threshold(gray, 20,255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)

        dilated=cv2.dilate(thresh,kernel,iterations=1)

        contours,_=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        rectsUsed = []
        for contour in contours:
            rects.append(cv2.boundingRect(contour))
            rectsUsed.append(False)
        def getXFromRect(item):
            return item[0]
        rects.sort(key = getXFromRect)
        newRects = []
        xThr = 25
        for supIdx, supVal in enumerate(rects):
            if (rectsUsed[supIdx] == False):
                currxMin = supVal[0]
                currxMax = supVal[0] + supVal[2]
                curryMin = supVal[1]
                curryMax = supVal[1] + supVal[3]
                rectsUsed[supIdx] = True
                for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):
                    candxMin = subVal[0]
                    candxMax = subVal[0] + subVal[2]
                    candyMin = subVal[1]
                    candyMax = subVal[1] + subVal[3]
                    if (candxMin <= currxMax + xThr):
                        currxMax = candxMax
                        curryMin = min(curryMin, candyMin)
                        curryMax = max(curryMax, candyMax)
                        rectsUsed[subIdx] = True
                    else:
                        break
                newRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])

        for rect in newRects:
            img = cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)
            cv2.rectangle(mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), -1)

    mask = cv2.inRange(mask,20,255)
    mask = mask * BW(frame)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    mask = cv2.inRange(mask,20,255)
    cv2.imshow("frame", frame)
    cv2.imshow("dilated", dilated)
    cv2.imshow("mask2",mask)
    cv2.waitKey(1)

cv2.waitKey(0)
print("how's it")

mask2 = cv2.inRange(mask, 100,255)
cv2.destroyAllWindows() 