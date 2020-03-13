import numpy as np
import cv2
from datetime import datetime


# draw lines along the points
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

#Load camera saved data
with np.load('camera.npz') as X:
      mtx,dist,_,_=[X[i] for i in ('mtx','dist','rvecs','tvecs')]


# ORB and BFMatcher objects
orb=cv2.ORB_create()
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# Model surface properties
model=cv2.imread('chess.jpg',0)
kp_m,ds_m=orb.detectAndCompute(model,None)

# Capture Video Object
cap = cv2.VideoCapture(0)





criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# arrays to store the object points in worls space
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# data points
vertices = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Operations on the frame
    kp_f,ds_f=orb.detectAndCompute(frame,None)

    # Match descriptors
    matches=bf.match(ds_m,ds_f)
    matches=sorted(matches,key=lambda x:x.distance)

    print(len(matches))
    if len(matches)>20:
          
          print('match')
          ret,corners=cv2.findChessboardCorners(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(7,6),None)
          if ret==True:
                print('True')
                corners2=cv2.cornerSubPix(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),corners,(11,11),(-1,-1),criteria)
                

                # Find rotation and translation vectors
                _, rvecs, tvecs, inliers  = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                # Project points to image plane
                imgpts,jac=cv2.projectPoints(vertices,rvecs,tvecs,mtx,dist)
                
                frame=draw(frame,corners,imgpts)
                filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
                cv2.imwrite('/sample_pictures'+filename,frame)

         

    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # undistort
    frame= cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    frame = frame[y:y+h, x:x+w]
    #Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
              break

# Release the capture
cap.release()
cv2.destroyAllWindows()
