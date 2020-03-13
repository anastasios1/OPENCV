import numpy as np
import cv2


image1=cv2.imread('logo.jpg',cv2.IMREAD_GRAYSCALE)        #quarry image
image2=cv2.imread('xmen.jpg',cv2.IMREAD_GRAYSCALE)       # train image

# Initiate ORB detector
orb=cv2.ORB_create()

# find keypoints and descriptors with ORB
key_points1,descriptors1=orb.detectAndCompute(image1,None)
key_points2,descriptors2=orb.detectAndCompute(image2,None)

# create BFMatcher object
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# Match Descriptors
matches=bf.match(descriptors1,descriptors2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first ten matches
image3=cv2.drawMatches(image1,key_points1,image2,key_points2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('FEATURE MATCHING',image3)

