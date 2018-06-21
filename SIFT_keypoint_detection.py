import numpy as np
import cv2

filename = 'pip6.jpg'


img = cv2.imread(filename)


gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray,None)


img3=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img4=cv2.drawKeypoints(gray,kp,img3)


cv2.imshow("SIFT1",img4)



