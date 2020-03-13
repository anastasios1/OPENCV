import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(1):
      # Capture each frame
      _,frame=cap.read()

      # Convert BGR to HSV
      hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

      # define range of blue color in HSV
      lower_blue=np.array([110,50,50])
      upper_blue=np.array([130,255,255])

      # Threshold the HSV image to get blue colors
      mask=cv2.inRange(hsv,lower_blue,upper_blue)

      # Bitwise AND mask and original image
      res=cv2.bitwise_and(frame,frame,mask=mask)

      cv2.imshow('FRAME',frame)
      cv2.imshow('MASK',mask)
      cv2.imshow('RESULT',res)

      k=cv2.waitKey(5) & 0XFF
      if k==27:
            break
cv2.destroyAllWindows()
cap.release()