import cv2
import math
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_PIXEL_AREA = 50
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 2

MAX_NORMAL_DISTANCE_BETWEEN_CHARS=8
MAX_ANGLE_BETWEEN_CHARS=20

class character:
    def __init__(self,contour):
        self.contour = contour
        self.boundingRectX,self.boundingRectY,self.width,self.height = cv2.boundingRect(self.contour)

        self.boundingRectArea = self.width * self.height

        self.centerX = (self.boundingRectX + self.boundingRectX + self.width) / 2
        self.centerY = (self.boundingRectY + self.boundingRectY + self.height) / 2

        self.diagonalSize = math.sqrt((self.width ** 2) + (self.height ** 2))

        self.aspectRatio = self.width / self.height
    def __eq__(self,other):
        for i in ['boundingRectX','boundingRectY','height','width']:
            if self.__getattribute__(i) == other.__getattribute__(i):
                return False
        return True
    def isValid(self):
        if self.boundingRectArea > MIN_PIXEL_AREA:
            if self.width > MIN_PIXEL_WIDTH:
                if self.height > MIN_PIXEL_HEIGHT:
                    if self.aspectRatio > MIN_ASPECT_RATIO:
                        if self.aspectRatio < MAX_ASPECT_RATIO:
                            return True
        return False
    def distance(self,other,normalized=True):
        dis=math.sqrt((self.centerX-other.centerX)**2
                     +(self.centerX-other.centerX)**2)
        if normalized:
            return dis/self.diagonalSize
        else:
            return dis

    def differenceAngle(self,other):
        return math.atan2(abs(self.centerY-other.centerY),
                          abs(self.centerX-other.centerX))\
               *(180/math.pi)
    def isNearChar(self,other):
        if self.distance(other)>MAX_NORMAL_DISTANCE_BETWEEN_CHARS:
            return False
        if self.differenceAngle(other)>MAX_ANGLE_BETWEEN_CHARS:
            return False
        if not 2/3<=self.boundingRectArea/other.boundingRectArea<=1.5:
            return False
        if not 5/9<=self.width/other.width<=1.8:
            return False
        if not 5/6<=self.height/other.height<=1.2:
            return False
        return True
    def findNearChars(self,listOfChars):
        nearChars=[]
        farChars=[]
        while listOfChars:
            currentChar=listOfChars.pop()
            if self.isNearChar(currentChar):
                nearChars.append(currentChar)
            else:
                farChars.append(currentChar)
        return nearChars,farChars
