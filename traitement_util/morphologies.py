import os.path
import cv2
import numpy as np
from .util_misc import *
def erosion(img):
    elem_struct = np.ones((3, 3), np.uint8)
    img_dilate = cv2.erode(img, elem_struct, iterations=1)
    return img_dilate

def dilatation(img, ksize=3):
    elem_struct = np.ones((ksize, ksize), np.uint8)
    img_dilate = cv2.dilate(img, elem_struct, iterations=3)
    return img_dilate

def ouverture(img, size=3):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))

