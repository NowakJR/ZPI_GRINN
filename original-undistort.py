# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:29:01 2017

@author: ondrej

Script undistorts the input image. It uses the provided calibration parameters.
"""

import cv2 as cv2
import pickle

paramsPath = 'Insert_path_to_calibration_parameters.pckl'

image_path = 'Insert_input_image_path_for_calibration.jpg'
image_pathCalib = 'Insert_output_image_path_for_calibration/calibrated.jpg'

f = open(paramsPath,'rb')
cab = pickle.load(f)
f.close()

mtx, dist, rvecs, tvecs = cab[:]

cv2.useOptimized()

img = cv2.imread(image_path)


h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(image_pathCalib, dst)