# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 12:28:24 2017

@author: ondrej kunte

Script computes calibration parameters from provided sets of chessboard images
Uses "pickle" serialization to save obtained calibration parameters in a 
form, which is usable for further processing.
"""

import numpy as np
import cv2
import glob
import pickle


cv2.useOptimized()
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# setup of captured chess-board size (WIDTH x HEIGHT)
BOARD_SIZE = (9,6)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((BOARD_SIZE[1]*BOARD_SIZE[0],3), np.float32)
objp[:,:2] = np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('insert_path_to_chessboard_images_direction\\*.jpg')
#images = images[:1]
counter = 0
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)



for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE ,None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        counter += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, BOARD_SIZE, corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(200)
        print(fname)

cv2.destroyAllWindows()
print('Number of photos processed: ', counter,)

#camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#cal = calibParams(mtx, dist, rvecs, tvecs)
cal = [mtx, dist, rvecs, tvecs]
points = [objpoints, imgpoints]
f = open('Insert_path_to_save_calibration_parameters\\calibParams.pckl','wb')
pickle.dump(cal, f)
f.close()

f = open('Insert_path_to_save_reference_parameters\\points.pckl','wb')
pickle.dump(points, f)
f.close()










