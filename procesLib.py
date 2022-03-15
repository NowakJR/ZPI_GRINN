# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 13:45:41 2017

@author: ondrej

Contains definitions of core processes for inspection 
Defines the class: Board()
"""

import cv2 as cv2
import numpy as np
import math
import operator


'''
Excepts image in arbitrary format and performs bilateral filtering in given setup
'''
def filter_bilateral(img, iter=5,kernel=5, area_effect=15, range_effect=3):
    
    for i in range(0,iter):
        blur = cv2.bilateralFilter(img, kernel, area_effect, range_effect)
        img = blur
    return img
        
def drawLines(img, lines):
    img_line = img.copy()
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img_line,(x1,y1),(x2,y2),(0,0,255),2)
    
    return img_line

def printImg(img):
    cv2.namedWindow('window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('window', img)
    cv2.waitKey(0)
    
def flatLight(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img[:,:,0] = 128
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

#------------------------------------------------------------------------------
"""
Created on Wed Sep 20 09:58:47 2017

@author: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc

"""



def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simpleWhiteBal(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        print ("Lowval: ", low_val)
        print ("Highval: ", high_val)

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

#------------------------------------------------------------------------------
"""
General board objects description
class Board
class PartContainer
class Part
"""
"""
class Board contains all information about the board (parts, pads, locations etc.)
"""

class Board:
    def __init__(self, image):
        self.partContainers = []
        self.image = image
        self.solderMaskColor= (np.zeros(3), np.zeros(3)) #low, high threshold
        self.partsCount = 0
        
    def addPart(self, container):
        self.partContainers.append(container)
        self.partsCount = len(self.partContainers)
        
    def removeLastPart(self):
        self.partContainers.pop()
        self.partsCount = len(self.partContainers)
        
    def getSubImage(self, container):
        x, y = container.nullpoint
        w, h = container.roi
        return self.image[y:y+h, x:x+w]  
    
    def getContainerByIdx(self, idx):
        for container in self.partContainers:
            if container.idx == idx:
                return container

        

"""
class PartContainer contains all information about the place of part
(roi, nullpoint and idx of container)
"""
class PartContainer:
    def __init__(self, roi, nullpoint, idx, part):
        self.roi = roi
        self.nullpoint = nullpoint
        self.idx = idx
        self.part = part
        
    def getPartRelativePosition(self):
        return tuple(map(operator.sub, self.part.position, self.nullpoint))

        

"""
class Part contains all information about the part
(partType, polarity, significant color)
partType -> string
polarity -> 0,1,2,3,4 = none,North,East,West,South
color = tuple(BGR_threshold, HSV_threshold, CIELab_threshold) ref to colorLib
"""
class Part:
   
    def __init__(self, partType, polarity, color, position, rotation):
        self.partType = partType
        self.polarity = polarity
        self.position = position #absolute position
        self.rotation = rotation
