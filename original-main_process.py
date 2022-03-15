# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:04:24 2017

@author: ondrej kunte
"""
import cv2 as cv2
import procesLib as lib
import colorLib as clib
import numpy as np
import pickle, os

'''Controlled image'''
image = None
imageFiltered = None
'''Reference board from library'''
reference = None

PIXEL_SIZE = 0.0016
MARK_DISTANCE_RATE = 0.8

'''MAX allowed rotation and distance thresholds'''
MAX_ERR_DIST = 10 #mm
MAX_ERR_ROT = 1 #degrees

'''General kerner for all morphological transformations'''
M_KERNEL_3x3 = np.ones((3,3), np.uint8)
M_KERNEL_5x5 = np.ones((5,5), np.uint8)
                
def loadImage(name):
    global image
    image = cv2.imread(name)
       

def loadReferenceBrd(name):
    path = os.getcwd()+"\\data\\boards\\"+name+".pkl"
    return pickle.load(open(path, "rb"))
  
    
def getSubImage(img, roi, nullpoint):
    return img[nullpoint[1]:nullpoint[1]+roi[1], nullpoint[0]:nullpoint[0]+roi[0]]
    
def getEucleidianDist(XYpos1, XYpos2):
    return np.linalg.norm(XYpos1 - XYpos2)

def thresholdByColor(imgBGR, colorKey):
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2Lab)
    #thresholding
    maskBGR = cv2.inRange(imgBGR, clib.colorPallete[colorKey].BGR['min'],
              clib.colorPallete[colorKey].BGR['max'])
    maskHSV = cv2.inRange(imgHSV[:,:,0], clib.colorPallete[colorKey].Hsv['min'],
              clib.colorPallete[colorKey].Hsv['max'])
    maskLAB = cv2.inRange(imgLAB[:,:,1:], clib.colorPallete[colorKey].lAB['min'],
              clib.colorPallete[colorKey].lAB['max'])
    
    #thresholdBGR && (thresholdHSV || thresholdLAB)
    return cv2.bitwise_and(cv2.bitwise_or(maskHSV, maskLAB), maskBGR)

def segment(image, foregroundMarkers, backgroundMarkers):
    #eroding setpoints
    foregroundMarkers = cv2.erode(foregroundMarkers, M_KERNEL_3x3, iterations = 1)
    backgroundMarkers = cv2.erode(backgroundMarkers, M_KERNEL_3x3, iterations = 1)
    
    #marking
    markers = backgroundMarkers
    markers[backgroundMarkers==255] = 1
    markers[foregroundMarkers==255] = 2
    markers = markers.astype('int32')
    
    #filter
    image = lib.filter_bilateral(image, iter=1, kernel=11,area_effect=15)
    
    #segmenting
    segmented_img = cv2.watershed(image, markers)
    
    #transfer to binary images
    segmented_binary = np.zeros(segmented_img.shape, dtype='uint8')        
    segmented_binary[segmented_img==2] = 255
    
    return segmented_binary

def getPosAndRot(segmentedImage, nullpoint):
    position = 0
    rotation = 0
    _, contours, _ = cv2.findContours(segmentedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(contours):
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(subImageBGR,[box],0,(255,0,0),3, cv2.LINE_AA)
        
        position = nullpoint + np.array(rect[0])
        rotation = rect[2]
    
    return position, rotation

def getTruePolarity(position_mark, position_part):
    diff = position_mark - position_part
    if (np.abs(diff[0]/diff[1]) < MARK_DISTANCE_RATE):
        abs_diff = np.abs(diff)
        if (abs_diff[0] > abs_diff[1]):
            #main x axis
            if (diff[0] <= 0):
                return 3
            else:
                return 4
        else:
            if (diff[1] <= 0):
            #main y axis
                return 2
            else:
                return 1
    else:
        #circular mark
        if (diff[0] <= 0 and diff[1] <= 0):
            return 3
        elif (diff[0] > 0 and diff[1] <= 0):
            return 2
        elif (diff[0] <= 0 and diff[1] > 0):
            return 1
        elif (diff[0] > 0 and diff[1] > 0):
            return 4
        else:
            return 0

def boardCheck():
    global reference, imageFiltered, MAX_ERR_DIST, MAX_ERR_ROT
    faultList = []
    
    #--------------------------------------------------------------------------
    #For every part container do inspection
    #--------------------------------------------------------------------------
    for containerIdx in range(1, reference.partsCount+1):
        #----------------------------------------------------------------------
        #Looking for correct type of part
        #----------------------------------------------------------------------
        
        maxMagnitude = 0
        nullpoint = reference.getContainerByIdx(containerIdx).nullpoint
        roi = reference.getContainerByIdx(containerIdx).roi
        subImageBGR = getSubImage(imageFiltered, roi, nullpoint)
        
        for color in clib.partColorKeys:
            mask = thresholdByColor(subImageBGR, color)
            magnitude = cv2.countNonZero(mask)
            
            if (magnitude > maxMagnitude):
                part = color
                maxMagnitude = magnitude
                maskBest = mask
        print(containerIdx, part, reference.getContainerByIdx(containerIdx).part.partType)
        
        if (part != reference.getContainerByIdx(containerIdx).part.partType):
            faultList.append([containerIdx, 'part_switch'])
            if (part == clib.colorPallete['empty']):
                faultList.append([containerIdx, 'missing part'])
            
        
        #----------------------------------------------------------------------
        #Looking for correct position and rotation
        #----------------------------------------------------------------------
        
        background = thresholdByColor(subImageBGR, 'solderMask')
        
        #Apply watershed segmentation to get foreground od component
        segmented_part = segment(getSubImage(image, roi, nullpoint), maskBest, background)
        
        #Get position and rotation of the component
        position_part, rotation_part = getPosAndRot(segmented_part, nullpoint)
        
        err_dist = getEucleidianDist(position_part,
                    reference.getContainerByIdx(containerIdx).part.position)*PIXEL_SIZE
        
        #get rotation and translation errors
        ref_rotation = reference.getContainerByIdx(containerIdx).part.rotation
        #solve 90degrees shift
        err_rot_list = [np.abs(rotation_part-ref_rotation-90), np.abs(rotation_part-ref_rotation+90),
                        np.abs(rotation_part-ref_rotation)]
        err_rotation = np.min(err_rot_list)

        #update error list
        if (err_dist > MAX_ERR_DIST):
            faultList.append([containerIdx, 'wrong possition'])
            
        if (err_rotation > MAX_ERR_ROT):
            faultList.append([containerIdx, 'rotated part'])
        
        print("Rotation err: {0}°, \nDistance err: {1} mm".format(err_rotation, err_dist))
        print()
        
        #----------------------------------------------------------------------
        #Looking for correct polarity
        #----------------------------------------------------------------------
        ref_polarity = reference.getContainerByIdx(containerIdx).part.polarity
        if (ref_polarity != 0):
            img = getSubImage(image, roi, nullpoint)
            imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgGrey = cv2.medianBlur(imgGrey,5)

            circles = cv2.HoughCircles(imgGrey,cv2.HOUGH_GRADIENT,1,50,
                            param1=100,param2=60,minRadius=10,maxRadius=100)

            circles = np.uint16(np.around(circles))
            if (not circles): 
                for color in clib.polarityKeys:
                    mask = thresholdByColor(img, color)
                    magnitude = cv2.countNonZero(mask)
            
                    if (magnitude > maxMagnitude):
                        maxMagnitude = magnitude
                        maskBest = mask
                        
                #Apply watershed segmentation to get polarity mark
                segmented_mark = segment(getSubImage(image, roi, nullpoint), maskBest, background)
                
                #Get position of the mark
                position_mark,_ = getPosAndRot(segmented_mark, nullpoint)
                
                true_polarity = getTruePolarity(position_mark, position_part)
                
                if (true_polarity != ref_polarity):
                    faultList.append([containerIdx, 'Incorrect polarity'])



if __name__ == "__main__":
    #set image path
    #imageName = 'insert_path_to_image.jpg'
    imageName = 'ref_board.jpg'
    loadImage(imageName)

    #filtering
    imageFiltered = lib.filter_bilateral(image)
    
    name = "ref_board"
    #name = "NX_board"
    reference = loadReferenceBrd(name)
    
    flist = boardCheck()

    cv2.destroyAllWindows()