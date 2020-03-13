import numpy as np
import cv2
from matplotlib import pyplot as plt

image=cv2.imread('spiderman_0.jpg')

#initiate ORB object
orb=cv2.ORB_create()

#find the keypoints in the ORB
key_points=orb.detect(image,None)

#compute the descriptors with ORB
key_points,des=orb.compute(image,key_points)

#draw the keypoints
image2=cv2.drawKeypoints(image,key_points,None,color=(0,255,0))

plt.imshow(image2)
plt.show()

cv2.imshow('KeyPoints',image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


