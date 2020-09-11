from plateRecognizer import detectPlate
from os import listdir
from os.path import join
import cv2
import numpy as np
DATA_DIR='imgs'

if __name__ == '__main__':
    imgs=[i for i in listdir(DATA_DIR) if i.split(".")[-1].lower() in ["jpg","png"]]
    # imgs=['testImg.jpg','new_car.jpg']
    allplates=[]
    for imageName in imgs:
        image = cv2.imread(join(DATA_DIR,imageName))
        plates = detectPlate(imageName,image)
        cv2.destroyAllWindows()
        print(f"{imageName} -> {plates}")
