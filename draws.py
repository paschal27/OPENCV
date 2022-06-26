import cv2 as cv
from cv2 import FONT_HERSHEY_TRIPLEX
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)
# img = cv.imread('Photos/cat.jpg')
# cv.imshow('Cat', img)

# blank[200:300,300:400] = 255,0,0
# cv.imshow('Green', blank)

#Rectangle
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=cv.FILLED)
cv.imshow('Rec', blank)

#Circle
cv.circle(blank, (250,250),40, (255,0,0), thickness=-1)
cv.imshow('Circle', blank)

#Line
cv.line(blank, (0,0),(250,250),(0,0,255),thickness=2)
cv.imshow('Line', blank)

#Text
cv.putText(blank, 'Hellllo', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), thickness=2)
cv.imshow('Text' , blank)
cv.waitKey(0)