import cv2
import math
import numpy as np

from . import utils

MIN_DISTANCE=0.3
HEIGHT_PADDING=1.5
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
class plate:
    def __init__(self, chars, image):
        self.chars=chars
        chars.sort(key = lambda currentChar: currentChar.centerX)
        self.centerX=int(chars[0].centerX+chars[-1].centerX)//2
        self.centerY=int(chars[0].centerY+chars[-1].centerY)//2
        self.center=(self.centerX,self.centerY)
        self.width= chars[-1].width/2+chars[0].width/2
        self.width+=chars[-1].centerX-chars[0].centerX
        self.width=int(self.width)

        self.height=int(sum(c.height for c in chars)/len(chars)*HEIGHT_PADDING)

        rotationAngle=math.asin((chars[-1].centerY-chars[0].centerY)/chars[0].distance(chars[-1],normalized=False))
        rotationMatrix=cv2.getRotationMatrix2D(self.center,
                                               rotationAngle*180/math.pi,
                                               1)
        h,w,channels=image.shape

        imgRotated = cv2.warpAffine(image, rotationMatrix, (w,h))       # rotate the entire image

        self.imgPlate = cv2.getRectSubPix(imgRotated,
                                          (self.width,self.height),
                                          self.center)
        self.imgValue,self.imgThresh=utils.imagePreprocess(self.imgPlate)

        # increase size of plate image for easier viewing and char detection
        self.imgThresh = cv2.resize(self.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
        # threshold again to eliminate any gray areas
        _, self.imgThresh = cv2.threshold(self.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    def isValid(self):
        charsInPlateArea=[]
        # check if all of the characters fits in the plate or not
        # and ommit chars which exceeded the boundaries
        for c in self.chars:
            if self.centerY+self.height/2>c.centerY+c.height/2:
                if self.centerY-self.height/2<c.centerY-c.height/2:
                    charsInPlateArea.append(c)

        # we want to delete smaller chars if they are so near
        charsInPlateArea.sort(key=lambda c:c.boundingRectArea)
        self.chars=[]
        while charsInPlateArea:
            currentChar=charsInPlateArea.pop(0)
            if charsInPlateArea==[] or min([currentChar.distance(i) for i in charsInPlateArea])>MIN_DISTANCE:
                self.chars.append(currentChar)

        self.chars.sort(key= lambda c:c.centerX)

        return len(self.chars)>3

    def refindAllCharacters(self):
        pass
    def recognizeChars(self,imgThresh,KNNmodel):
        self.chars.sort(key=lambda c:c.centerX)
        strChars=''
        for currentChar in self.chars:
            currentChar.imgThresh = imgThresh[currentChar.boundingRectY : currentChar.boundingRectY + currentChar.height,
                                              currentChar.boundingRectX : currentChar.boundingRectX + currentChar.width]

            currentChar.imgThresh=cv2.resize(currentChar.imgThresh,
                                             (RESIZED_CHAR_IMAGE_WIDTH,
                                             RESIZED_CHAR_IMAGE_HEIGHT))\
                                  .reshape((1,-1))\
                                  .astype(np.float32)

            res=int(KNNmodel.findNearest(currentChar.imgThresh, k = 1)[1][0][0])
            strChars+=chr(res)
        return strChars
