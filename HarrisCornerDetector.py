import numpy as np
import cv2

#load image
image=cv2.imread('chessboard.jpg')

#transform to gray
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#transform to float32
gray=np.float32(gray)

#apply corner Harris detection
dst=cv2.cornerHarris(gray,2,3,0.04)
dst=cv2.dilate(dst,None)

#apply the threshold
image[dst>0.01*dst.max()]=[0,0,255]

#display the image
cv2.imshow('IMAGE',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
