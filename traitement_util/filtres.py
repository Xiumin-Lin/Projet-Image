import os
import json
import os.path
import re
import sys
from .util_misc import *
import matplotlib.pyplot as plt
import cv2
import numpy as np

def conversion_en_gris(img):
    img_gris = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # img value convert to int prevent the error : "overflow encountered in ubyte_scalars"
            gray_value = int((int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2])) / 3)
            for k in range(3):
                img_gris[i, j] = gray_value

    # OU Version de CV2
    # img_gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gris

def egaliser(img, histo):
    """
    Opération d'égalisation de l'image donnée en param
    :param img l'image à égaliser
    :param histo l'historigramme de l'image
    :return l'image égalisée
    """
    h_cumule, cumul = histo_cumule(histo)
    img_egalise = np.zeros(img.shape, dtype=np.uint8)
    n = len(histo)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_value = ((n / cumul) * h_cumule[img[i][j]]) - 1
            # img_egalise = max(0, newValue)
            if new_value > 0:
                img_egalise[i][j] = new_value
            else:
                img_egalise[i][j] = 0
    return img_egalise

def convolution_2d(img, noyau):
    """ Convolution avec un noyau à 2 dimensions """
    img_convolve_2d = cv2.filter2D(img, ddepth=-1, kernel=noyau)
    return img_convolve_2d

def filtre_moyenneur(img):
    moy_kernel_5x5 = np.ones((5, 5), np.uint8) / 25
    img_moyenneur = convolution_2d(img, moy_kernel_5x5)
    return img_moyenneur

def filtre_median(img, ksize=3):
    img_median = cv2.medianBlur(img, ksize=ksize)
    return img_median

def filtre_gaussian(img, ksize=17):
    # gaus_kernel_3x3 = np.asarray([[1, 2, 1],
    #                               [2, 3, 2],
    #                               [1, 2, 1], ], dtype=np.uint8) / 16
    #
    # gaus_kernel_5x5 = np.asarray([[1, 4, 5, 4, 1],
    #                               [4, 16, 26, 16, 4],
    #                               [7, 26, 41, 26, 7],
    #                               [4, 16, 26, 16, 4],
    #                               [1, 4, 5, 4, 1]], dtype=np.uint8) / 273
    # img_gaussien = convolution_2d(img, gaus_kernel_5x5)

    img_gaussien = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img_gaussien

def filtre_sobel(img):
    img_sobel = cv2.Sobel(img, ddepth=-1, dx=1, dy=1, ksize=1)
    return img_sobel


# non utilisée
###########################################
def convolution_diy(img, noyau):
    img_convolve = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    # Test que le noyeau est de (n x n) avec n impair
    if (len(noyau) != len(noyau[1])) or (len(noyau) % 2 == 0):
        return img_convolve

    # normalisation du noyeau
    centre_noyau = int((len(noyau) - 1) / 2)
    moy_noyau = 0

    for i in noyau:
        for j in i:
            moy_noyau += abs(j)
    print(moy_noyau)

    for pixelLine in range(len(img_convolve)):
        for pixel in range(len(img_convolve[pixelLine])):
            ie = 0
            for ligneN in range(len(noyau) - 1):
                for numLigneN in range(len(noyau[ligneN]) - 1):
                    if (not (((pixel - (numLigneN - centre_noyau)) < 0) or (
                            (pixelLine - (ligneN - centre_noyau)) < 0))):
                        print(f"convolution de pixel {pixelLine}:{pixel}, "
                              f"{(pixel - (ligneN - centre_noyau))}:{(pixelLine - (ligneN - centre_noyau))}")
                        e = img[pixelLine - (ligneN - centre_noyau)][pixel - (numLigneN - centre_noyau)]

                    else:
                        print(f"pixel skippé: {pixelLine}{pixel}")
            ie = ie / moy_noyau
            img_convolve[pixelLine][pixel] = ie
    return img_convolve
###########################################