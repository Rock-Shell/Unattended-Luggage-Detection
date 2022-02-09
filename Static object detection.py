import cv2
import numpy as np 
import os
import time
import cirle
import math

import sys
sys.path.insert(1, "D:/Intership work/Anemoi assignment/left luggage detection/main/Yolov5")
import Yolov5
from Yolov5 import yolov5

def blur(img):
    #return cv2.medianBlur(img, 5)
    #return cv2.bilateralFilter(img, 9, 75,75)
    return cv2.GaussianBlur(img, (5,5),-1)

def BW(img): # Black and White
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ED(a,b,c,d):
    d = math.sqrt((c-a)**2 + (d-b)**2)
    return d

def BoxOverlap(a,b):
    return (abs((a[0] + a[2]/2) - (b[0] + b[2]/2)) * 2 <= (a[2] + b[2])) and (abs((a[1] + a[3]/2) - (b[1] + b[3]/2)) * 2 <= (a[3] + b[3]))

path = "D:/Intership work/Anemoi assignment/left luggage detection/dataset/S1-T1-C/S1-T1-C/video\pets2006/S1-T1-C/3"
path2 = "D:/Intership work/Anemoi assignment/left luggage detection/main"
base = "S1-T1-C.00000"
start = time.time()

FF = cv2.imread(path+'/' + base + ".jpeg")
white = 255*np.ones(FF.shape[:2], np.uint8)
black = np.zeros(FF.shape[:2], np.uint8)
mask2 = 255*np.ones(FF.shape[:2], np.uint8)
moving = np.zeros(FF.shape,np.uint8)
kernel2 = np.ones((5,5),np.uint8)
kernel3 = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

background = cv2.imread(path2 + "/" + "background/background3.jpeg")
# background2 = cv2.absdiff(FF, background)
# background2 = np.float32(background2)

j=-1
frame2 = FF.copy()
###################
mask2 = np.zeros_like(FF)
mask2[...,1]=255
###################
k =0 # MPR
for i in os.listdir(path):
    j+=1
    now = time.time()
    frame = cv2.imread(path + "/" + i)

    ############# Back ground Subtraction (1)##############
    mask = cv2.absdiff(BW(frame), BW(background))
    ###################################################
    if j%30 == 0:
        red = 0
        moving = cv2.absdiff(BW(frame), BW(frame2))

        ###########
        # using optic flow algorithnm to detect moving object
        # flow = cv2.calcOpticalFlowFarneback(BW(frame2),BW(frame), None,0.5, 3, 15, 3, 5, 1.2, 0)
        # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # mask2[..., 0] = angle * 180 / np.pi / 2
        # mask2[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        # # Converts HSV to RGB (BGR) color representation 
        # rgb = cv2.cvtColor(mask2, cv2.COLOR_HSV2BGR)
        #############

        ######### Moving object detection using consecutive frame difference (2)
        frame2 = frame.copy()

        gray = moving.copy() #BW(moving)
        gray = blur(blur(blur(gray)))
        gray =  3*gray
        _, thresh = cv2.threshold(gray, 20,255, cv2.THRESH_BINARY)
        # copy = thresh.copy()
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)

        thresh=cv2.dilate(thresh,kernel,iterations=1)
        copy = mask.copy()

        contours,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x,y,w,h)=cv2.boundingRect(contour)
            # if cv2.contourArea(contour)<100:
                # cv2.fillPoly(mask, contour, (0,0,0))
                # continue
            cv2.rectangle(mask,(x,y),(x+w,y+h),(0,0,0),-1)
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) # red
        #################################################################

        ##### Probable regions for left luggage (3)############
        mask = blur(blur(mask))
        # mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3)
        mask = cv2.inRange(mask,20,255)
        copy = mask.copy()

        cont2,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame,cont2,-1, (255,255,0),2)
        cont2 = sorted(cont2, key=lambda x: cv2.contourArea(x),reverse=True)

        Output = np.zeros(frame.shape,np.uint8)
        init_roi = []
        for c in range(len(cont2)):
            if cv2.contourArea(cont2[c])<100:
                cv2.fillPoly(mask, [cont2[c]], (0,0,0))
                continue
            k+=1
            (x,y,w,h)=cv2.boundingRect(cont2[c])
            init_roi.append((x,y,w,h))
        #####################################################################

        ###### Combining close ROIs #############################
        if j == 0:
            cv2.imshow("mask",mask)
            cv2.imshow("frame",frame)
            continue

        roi = []
        sp = [0 for i in range(len(init_roi))]
        # '''
        for i in range(len(init_roi)):
            if sp[i] ==1:
                continue
            sp[i] = 1
            temp = [init_roi[i]]
            for j in range(1,len(init_roi)):
                if sp[j] == 1:
                    continue
                thresh_dist = (init_roi[i][2]+init_roi[j][2])//4 + 100
                # print(thresh_dist)
                if ED(init_roi[i][0],init_roi[i][1],init_roi[j][0],init_roi[j][1]) < thresh_dist:
                    sp[j] = 1
                    temp.append(init_roi[j])
            c = [0,0,0,0]
            for ki in temp:
                c[0] += ki[0]
                c[1] += ki[1]
                c[2] += ki[2]
                c[3] += ki[3]
            c[0],c[1],c[2],c[3] = c[0]//len(temp), c[1]//len(temp), c[2]//len(temp), c[3]//len(temp)
            roi.append(c)
        print(len(roi),end=" ")
        # '''

        ############# Applying model and finding hoomans #######

        humans,bags = yolov5.detect(frame)
        # humans, bags = [], []
        print(f"humans: {humans} bags: {bags}")

        humans, bags = np.array(humans), np.array(bags)
        # humans, bags = [[]], [[]] # if you don't want to apply model

        ''' no need in yolov5
        relative_x = lambda x,fc: (x * fc)//1280
        relative_y = lambda y,fc: (y * fc)//720

        if bags != [[]]:
            # frame = cv2.resize(frame, (1280,720))
            for i in range(len(bags)):
                # print(bag, frame.shape)
                bags[i][0] = relative_x(bags[i][0],frame.shape[1])
                bags[i][1] = relative_y(bags[i][1],frame.shape[0])
                bags[i][2] = relative_x(bags[i][2],frame.shape[1])
                bags[i][3] = relative_y(bags[i][3],frame.shape[0])
                # cv2.rectangle(frame, (bag[0],bag[1]),(bag[2],bag[3]), (255,0,0),2)
        
        if humans != [[]]:
            # frame = cv2.resize(frame, (1280,720))
            for i in range(len(humans)):
                # print(bag, frame.shape)
                humans[i][0] = relative_x(humans[i][0],frame.shape[1])
                humans[i][1] = relative_y(humans[i][1],frame.shape[0])
                humans[i][2] = relative_x(humans[i][2],frame.shape[1])
                humans[i][3] = relative_y(humans[i][3],frame.shape[0])
        '''

        for (x,y,w,h) in roi:
            out = np.zeros(frame.shape,np.uint8)
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1) # draw roi with red
            centroid = (2*x+w)//2,(2*y+h)//2
            out = cirle.roi(w, frame,centroid, out)

            ######### Hooman detection and roi rejection ############
            # give frame instead of roi to the model
            # check % overlap of roi and hooman
            # if % is like 90% then roi is of hooman and reject that roi.
            # however, if it is less like 60-70%, it might contain bag as well.

            for human in humans:
                h2 = human[0], human[1], human[2]-human[0], human[3]-human[1]
                centroid2 = (2*h2[0]+h2[2])//2,(2*h2[1]+h2[3])//2
                # if ED(centroid, centroid2) 
                if BoxOverlap(h2, (x,y,w,h)) == True or ED(*centroid, *centroid2) < 3*w:
                    out = np.zeros(frame.shape,np.uint8)
                    cv2.rectangle(frame, (human[0],human[1]),(human[2],human[3]), (255,0,0),2) # draw hoomans with blue
                    break
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1) # draw roi not overlapping with hooman with red
                red = x,y,w,h


            ###########################################
            Output = cv2.bitwise_or(Output,out)
            # cv2.waitKey(0)

        # cv2.drawContours(frame, cont2, 0, (0,255,0), 2)
        #####################################################

        cv2.imshow("mask",mask)
        cv2.imshow("frame",frame)
        cv2.imshow("output",Output)
        cv2.imwrite(f"{path2}/output/{j}.jpeg",Output)
        # print(k,end=' ')
    if red != 0:
        cv2.rectangle(frame,(red[0],red[1]),(red[0]+red[2],red[1]+red[3]),(0,0,255),1)
    cv2.imwrite(f"{path2}/output/{j}.jpeg",frame)

    cv2.waitKey(1)

cv2.waitKey(0)
print("how's it")
cv2.destroyAllWindows()