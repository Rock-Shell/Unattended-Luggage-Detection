import cv2
import numpy as np

def roi(W,img,centroid,mask):
    # img=cv2.imread('S1-T1-C.01416.jpeg',cv2.IMREAD_UNCHANGED)
    y=(W//2+100,W//2+25)
    z=(W//2+140,W//2+42)
    
    # cv2.drawMarker(	img, centroid, (0,0,255),cv2.MARKER_CROSS,25,1)

    # cv2.ellipse(mask,tup,y,0,0,360,(1,1,1),-1)
    cv2.ellipse(mask,centroid,z,0,0,360,(1,1,1),-1)
    # mask=img-f2
    mask=img*mask
    # mask=np.where(mask>0,255,0)
    mask = mask.astype(np.uint8)
    return mask


    # cv2.circle()
    # cv2.bitwise_xor()