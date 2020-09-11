import cv2
import numpy as np

from .Char import character
from .Plate import plate
from . import utils

from os.path import join,dirname

def detectPlate(imageName,image,resize_w=None):
    if resize_w!=None:# and resize_w<image.shape[0]:
        shape=image.shape
        image = cv2.resize(image, (resize_w, resize_w*shape[0]//shape[1]))
    imgValue,imgThresh=utils.imagePreprocess(image)
    possibleChars=utils.findCharacters(imgThresh)
    groupedCahrs=utils.groupChars(possibleChars)
    plates=[plate(group,image) for group in groupedCahrs]
    plates=[i for i in plates if i.isValid()]
    KNNmodel = cv2.ml.KNearest_create()
    classes = np.loadtxt(join(dirname(__file__),
                              "classifications.txt"),
                         np.float32).reshape((-1,1))
    flattenImages = np.loadtxt(join(dirname(__file__),
                                    "flattened_images.txt"),
                                np.float32)
    KNNmodel.setDefaultK(1)
    KNNmodel.train(flattenImages, cv2.ml.ROW_SAMPLE, classes)
    return [p.recognizeChars(imgThresh,KNNmodel) for p in plates]
