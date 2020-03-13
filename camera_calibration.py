import numpy as np
import cv2
from datetime import datetime

#termination criteria
criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

#prepare object points
objp=np.zeros((6*7,3),np.float32)
objp[:,:2]=np.mgrid[0:7,0:6].T.reshape(-1,2)

#arrays to store object points and image points
objpoints=[]       #3d points in real world space
imgpoints=[]      #2d points in image plane


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    
    

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #find the chess borad corners
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY), (7,6), None)
    if ret==True:
          print(ret)
          #add object points and image points after refining them
          objpoints.append(objp)
          
          filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
          

          corners2=cv2.cornerSubPix(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),corners,(11,11),(-1,-1),criteria)
          imgpoints.append(corners2)

          # Draw and display the corners
          frame=cv2.drawChessboardCorners(frame,(7,6),corners2,ret)
          cv2.imwrite('data/'+filename,frame)
          # calibrate webcam and save output
          ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).shape[::-1], None, None)
          np.savez("camera.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
          

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
