# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 13:42:06 2017

@author: ondrej

Stores known color thresholds for different part packages and it's parts
Defines the class: color
"""
import numpy as np

class color():
    def __init__(self, bgr, hsv, lab):
        self.BGR = {'min': np.array(bgr[0]), 'max' : np.array(bgr[1])}
        self.Hsv = {'min': np.array(hsv[0]), 'max' : np.array(hsv[1])}
        self.lAB = {'min': np.array(lab[0]), 'max' : np.array(lab[1])}

solderMask = color(([60,88,77],[110,149,143]), (28, 70), ([107, 132],[124, 165]))
resBlack = color(([32,43,66],[92,89,105]),(0, 16),([134, 122],[142, 145]))
ceramicCapWhite = color(([161, 180,194],[194, 205, 216]), (9, 17), ([129, 133],[132, 138]))
ceramicCapLightBrown = color(([115,165,190],[133,171,197]),(20, 22),([128, 154],[131, 169]))
ceramicCapDarkBrown = color(([74,112,158],[104,129,171]), (12, 15), ([136, 146],[140, 159]))
tantalYellow = color(([36,159,206],[71,183,234]), (20, 22), ([133, 189],[140, 197]))
tantalYellowP = color(([17,120,200],[51,134,255]), (14, 17), ([150, 186],[158, 190]))
tantalBlack = color(([49,38,50],[68,59,72]), (145, 165), ([134, 121],[138, 126]))
tantalBlackP = color(([189,171,169],[212,181,180]), (120, 123), ([132, 114],[135, 116]))
LEDdiode = color(([174,191,216],[174,211,229]), (16, 22), ([127, 149],[130, 156]))
LEDdiodeP = color(([83,164,94],[131,195,174]), (41, 62), ([88, 147],[107, 177]))
diode = color(([12,80,221],[140,175,255]), (8,12), ([153, 155],[177, 192]))
diodePblue = color(([99,85,52],[143,101,83]), (108, 114), ([125, 98],[136, 107]))
integratedCirc = color(([48,38,49],[62,53,64]), (145, 172), ([134, 121],[135, 127]))
emty = color(([61,88,77],[110,150,143]), (28, 71), ([108, 132],[124, 164]))


colorPallete = {'solderMask' : solderMask,
                'resBlack': resBlack, 
                'ceramicCapWhite' : ceramicCapWhite,
                'ceramicCapLightBrown' : ceramicCapLightBrown,
                'ceramicCapDarkBrown' : ceramicCapDarkBrown,
                'tantalYellow' : tantalYellow,               
                'tantalBlack' : tantalBlack,                
                'LEDdiode' : LEDdiode,                
                'diode' : diode,               
                'integratedCirc' : integratedCirc,
                'empty': emty}

colorsKeys = colorPallete.keys()

partColorKeys = colorPallete.copy()
partColorKeys.pop('solderMask')

polarityPallete = {'tantalYellowP' : tantalYellowP,
                   'tantalBlackP' : tantalBlackP,
                   'LEDdiodeP' : LEDdiodeP,
                   'diodePblue' : diodePblue}

polarityKeys = polarityPallete.keys()

def HSVtransfer(hsv):
    return 180*hsv/255

