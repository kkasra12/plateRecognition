import cv2
import numpy as np
from .Char import character
def imagePreprocess(img):
    _,_,imgValue = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2HSV))
    kernel = np.ones((3, 3),np.uint8)
    imgTopHat = cv2.morphologyEx(imgValue, cv2.MORPH_TOPHAT, kernel)
    imgBlackHat = cv2.morphologyEx(imgValue, cv2.MORPH_BLACKHAT, kernel)
    imgGrayscalePlusTopHat = cv2.add(imgValue, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat,
                                                       imgBlackHat)
    imgBlurred = cv2.GaussianBlur(imgGrayscalePlusTopHatMinusBlackHat, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      19,
                                      9)

    return imgValue,imgThresh

def findCharacters(imgThresh):
    contours, npaHierarchy = cv2.findContours(imgThresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    possibleChars=[]
    for contour in contours:
        '''this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
        note that we are not (yet) comparing the char to other chars to look for a group'''
        possibleChar = character(contour)
        if possibleChar.isValid():
            possibleChars.append(possibleChar)
    return possibleChars

def groupChars(possibleChars):
    groupedCahrs=[]

    while possibleChars:
        currentChar=possibleChars.pop()
        currentGroup,possibleChars=currentChar.findNearChars(possibleChars)
        if len(currentGroup)>=2:
            currentGroup.append(currentChar)
            groupedCahrs.append(currentGroup)
        else:
            pass
            # print(len(currentGroup))
    return groupedCahrs
