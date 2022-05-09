import cv2
import numpy as np


def convolution_2d(img, noyau):
    """
    input: une [numpy.array] de l'image a convolve
    input: une [numpy.array] du noyeay
    output: une [numpy.array] de l'image après convolution
    
    
    Convolution avec un noyau à 2 dimensions
    """
    
    img_convolve_2d = cv2.filter2D(img, ddepth=-1, kernel=noyau)
    return img_convolve_2d


def filtre_moyenneur(img):
    """
    input: une [numpy.array] de l'image a laquelle il faut appliquer un filtre moyenneur
    output: une [numpy.array] de l'image après application du filtre
    
    
    applique un filtre moyenneur sur une image
    """
    moy_kernel_5x5 = np.ones((5, 5), np.uint8) / 25
    img_moyenneur = convolution_2d(img, moy_kernel_5x5)
    return img_moyenneur


def filtre_median(img, ksize=3):
    """
    input: une [numpy.array] de l'image a laquelle il faut appliquer un filtre median
    input: un [int] de la taille du noyeau pour le filtre median
    output:  une [numpy.array] de l'image après application du filtre median

    applique un filtre median sur une image
    """
    img_median = cv2.medianBlur(img, ksize=ksize)
    return img_median


def filtre_gaussian(img, ksize=17):
    """
    input: une [numpy.array] de l'image a laquelle il faut appliquer un filtre gaussien (convolution)
    input: un [int] de la taille du noyeau pour le filtre gaussien
    output:  une [numpy.array] de l'image après application du filtre gaussien
    
    applique un filtre gaussien sur une image
    """
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
    """
    input: une [numpy.array] de l'image a laquelle il faut appliquer un filtre de sobel (convolution)
    input: un [int] de la taille du noyeau pour le filtre de sobel
    output:  une [numpy.array] de l'image après application du filtre de sobel
    
    applique un filtre de sobel sur une image
    """
    img_sobel = cv2.Sobel(img, ddepth=-1, dx=1, dy=1, ksize=1)
    return img_sobel


# non utilisée
###########################################
def convolution_diy(img, noyau):
    """
    input: une [numpy.array] de l'image a laquelle il faut appliquer un filtre donné par le 2e input
    input:  une [numpy.array] du noyeau
    output:  une [numpy.array] de l'image après application de la convolution
    
    applique la convolution sur une image
    """
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
