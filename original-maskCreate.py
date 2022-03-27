# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:15:51 2017

@author: ondrej

Interactive process for creating the Golden board (reference)
"""

import cv2 as cv2
import numpy as np
from math import sqrt
import procesLib as lib
import pickle
import os
import colorLib as clib

drawing = False # true if mouse is pressed
ix,iy = -1,-1
firstClick = False
cornerIdx = 0
containerIdx = 1
partCount = 0
padIdx = 1
rectPoints = []
pads = []
cornerIdx = 0
image = None
imageOrig = None


imagePath = r'crop_1.jpg'


# mouse callback function
def getCornerPoints(event, x, y, flags, param):   
    global cornerIdx, rectPoints
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image, (x,y),5, (255,0,0), -1)
        cornerIdx = cornerIdx +1
        rectPoints.append((x,y))
        
# drawing rectangle function
def drawPart(image, channels, show, partID = 0, color=(0,0,255), box=[], rect=[]):
    if len(box) == 0:
        box = cv2.boxPoints(rect)
        box = np.int0(box) #rounding
    if channels > 1:
        cv2.drawContours(image,[box],0,color,3, cv2.LINE_AA)
    else:
        cv2.drawContours(mask_parts,[box],0,partID,-1)
    if show:
        cv2.imshow('image', image)
    cv2.waitKey(2)
    
# get minimal Eucleidian distance
def getMinDist(pad, coordinates):
    return sqrt((pad[0]-coordinates[0])**2 + 
                (pad[1]-coordinates[1])**2)

def getAverageColor(img):
    return [np.uint8(np.mean(img[:, :, i])) for i in range(img.shape[-1])]

    
    
def getNewPart():
    
    global rectPoints, padIdx, cornerIdx, image
    cornerIdx = 0
    while(1):
        #setting up the pads
        if cornerIdx >= 4:
    
            rect = cv2.minAreaRect(np.array(rectPoints))
            
            
            drawPart(image, 3, True, rect=rect)
            #drawPart(rect, mask_pads, 1, False, partID=partIdx)        
            #name = raw_input('Choose: res1206, capLight1206, cap0805, tantal, doide: ')
            
            rect = list(rect)
            
            while(1):
                partType = input('PartType - Choose from: \n'+repr(clib.colorsKeys))
                if partType in clib.colorPallete:
                    color = clib.colorPallete[partType]
                    break
                else:
                    print('Wrong name of the part! Choose again')
            
            
            polarity = input('Polarity - Choose: 0-none, 1-North, 2-South, 3-West, 4-East')
            
            position = list(rect[0])
            position = [int(x) for x in position]
            position = tuple(position)
            
            rotation = rect[2]

            part = lib.Part(partType, polarity, color, position, rotation)
            
            padIdx = padIdx + 1
            cornerIdx = 0
            rectPoints = []
            
            cv2.imshow('image',image)
            cv2.waitKey(1)
            
            return part
                               
        cv2.imshow('image',image)
        cv2.waitKey(1)

def getNewContainer(part):
    global rectPoints, containerIdx, cornerIdx, image, imageOrig
    cornerIdx = 0
    while(1):
        
        if cornerIdx >= 2:
    
            [x,y,w,h] = cv2.boundingRect(np.array(rectPoints)) #returns [x,y,w,h]
            box = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
            
            drawPart(image, 3, True, color=(0,255,0), box=box)
            
            nullpoint = (x, y)
            roi = (w, h)
            idx = containerIdx
            
            containerIdx = containerIdx +1
                      
            container = lib.PartContainer(roi, nullpoint, idx, part)
            
            cv2.imshow('image',image)
            cv2.waitKey(1)
            rectPoints = []
            return container
        
        cv2.imshow('image',image)
        cv2.waitKey(1)
    

# load image and prepare a window
def setImage():
    global image, imageOrig, mask_parts
    imageOrig = cv2.imread(imagePath)
    imageOrig = lib.filter_bilateral(imageOrig)
    image = np.copy(imageOrig)
    cv2.namedWindow('image')        
    cv2.setMouseCallback('image', getCornerPoints)
    mask_parts = np.zeros(image.shape[:2], np.uint16)


def saveBoard(brd, name):
    path = os.getcwd()+"\\data\\boards\\"+name+".pkl"
    pickle.dump(brd, open(path,"wb"))


# START of the process, get board info - parts and containers
def updateBoard(brd):
        newPart = getNewPart()
        container = getNewContainer(newPart)
        brd.addPart(container)
  
    
# ENTRY POINT of the application
if __name__ == "__main__":
    setImage()
    board = lib.Board(imageOrig)
    board.solderMaskColor = clib.solderMask
    while(1):
        updateBoard(board)
        k = input('Next part? (Y/N):')

        if k in ('N', 'n'):
            break

    name = input('Insert name of the board: ')
    saveBoard(board, name)
    del board
    cv2.destroyAllWindows()

