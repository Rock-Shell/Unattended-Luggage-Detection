import cv2
import numpy as np
import os

path = "D:/Intership work/Anemoi assignment/left luggage detection/dataset/S1-T1-C/S1-T1-C/video\pets2006/S1-T1-C/4"
path2 = "D:/Intership work/Anemoi assignment/left luggage detection/trial"
base = "S1-T1-C.00000"

'''
Backgrounds
1 - 1:1000
2 - 1-1000
3 - 1st 
4 - 700-1100

add weighted is causing unwanted noise. So background, should be created by cropping it from different frames.
'''


FF = cv2.imread(path+'/' + base + ".jpeg")
img = np.zeros(FF.shape)
# weight = 1/(len(os.listdir(path))/4)
weight = 0.01
# print(os.listdir(path)[0])
# print(len(os.listdir(path)[:100]))
# '''
j = 0
for i in os.listdir(path)[700:1100]:
        j+=1
        if j%4 != 0:
                continue
        if j % 100 == 0:
                print(j, end=" ")
        frame = cv2.imread(path+'/' +  i)
        img = img + weight* frame

img = img.astype(np.uint8)
cv2.imshow("background",img)
print()
print("press any key to save background")
cv2.waitKey(0)

cv2.imwrite(path2 + "/"+"background/background4.jpeg", img)
cv2.destroyAllWindows()
# '''