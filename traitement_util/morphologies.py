import functions
from .util_misc import *


def erosion(img, ksize=3):
    elem_struct = np.ones((ksize, ksize), np.uint8)
    img_dilate = cv2.erode(img, elem_struct, iterations=1)
    return img_dilate


def dilatation(img, ksize=3):
    elem_struct = np.ones((ksize, ksize), np.uint8)
    img_dilate = cv2.dilate(img, elem_struct, iterations=1)
    return img_dilate


def ouverture(img, size=3):
    # img_erode = erosion(img, size)
    # img_dilate = dilatation(img_erode, size)
    # return img_dilate
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
