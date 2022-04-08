from enum import Enum
from site import venv
import matplotlib.image as mplimg
from functions import conversion_en_gris
import numpy as np
import functions
import os
import matplotlib.pyplot as plt
# Enumère les différents type d'extension d'image
class ImageExtension(Enum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"


# TODO: pouvoir choisir entre base_test & base_validation
base_path = "res" + os.path.sep
# récupère les valeurs de ImageExtension
valide_extension = [item.value for item in ImageExtension]

# Compteur pour calculer le taux de reussite de l'algo
traitement_reussite = 0
nb_image_total = 0

# Détection pour chaque fichier image de la base selectionnée


def show_img(img, img_title):
    plt.figure()
    plt.title(img_title)
    plt.imshow(img, cmap=plt.cm.gray)
    

def main():
    filename = "99.png"
    img = ((mplimg.imread(base_path + filename).copy() * 255).astype(np.uint8))
    noy = np.array([[1,1,1,1,1,1,1,1,1,1,1], 
                    [1,1,1,1,1,1,1,1,1,1,1], 
                    [1,1,1,1,1,1,1,1,1,1,1], 
                    [1,1,1,1,1,1,1,1,1,1,1], 
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1]])
    for i in range(17, 25):
        img_cnvlvd = convolution_clr_diy(img, noy, i)
        show_img(img_cnvlvd, str(i))
        print(i)
    
    plt.show()
    show_img(img, "A")
    show_img(img_cnvlvd, "B")
    plt.show()
    print("A")
    

def convolution_diy(img, noyau):

    img_convolve = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    #Test que le noyeau est de (n x n) avec n impair
    if (len(noyau) != len(noyau[0])) or (len(noyau) % 2 == 0):
        return img_convolve
    
    #normalisation du noyeau
    centre_noyau = int((len(noyau) - 1) / 2)
    sum_values_kernel = 0

    for i in noyau:
        for j in i:
            sum_values_kernel += abs(j)
    print(sum_values_kernel)
    for pixelLine in range(len(img_convolve) - (len(noyau)//2)):
        for pixel in range(len(img_convolve[pixelLine]) - (len(noyau)//2)):
            ie = 0
            for ligneNoy in range(len(noyau) - 1):
                for caseNoy in range(len(noyau[ligneNoy]) - 1):
                    e = img[pixelLine - (ligneNoy - centre_noyau)][pixel - (caseNoy - centre_noyau)]
                    ie += e
            ie = ie / (16)
            img_convolve[pixelLine][pixel] = ie
    return img_convolve

def convolution_clr_diy(img, noyau, bi):

    img_convolve = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    #Test que le noyeau est de (n x n) avec n impair
    if (len(noyau) != len(noyau[0])) or (len(noyau) % 2 == 0):
        return img_convolve
    
    #normalisation du noyeau
    centre_noyau = int((len(noyau) - 1) / 2)
    sum_values_kernel = 0

    for i in noyau:
        for j in i:
            sum_values_kernel += abs(j)
    print(sum_values_kernel)
    for pixelLine in range(len(img_convolve) - (len(noyau)//2)):
        for pixel in range(len(img_convolve[pixelLine]) - (len(noyau)//2)):
            r = v = b = 0
            for ligneNoy in range(len(noyau) - 1):
                for caseNoy in range(len(noyau[ligneNoy]) - 1):
                    re = img[pixelLine - (ligneNoy - centre_noyau)][pixel - (caseNoy - centre_noyau)][0]
                    ve = img[pixelLine - (ligneNoy - centre_noyau)][pixel - (caseNoy - centre_noyau)][1]
                    be = img[pixelLine - (ligneNoy - centre_noyau)][pixel - (caseNoy - centre_noyau)][2]
                    
                    r += re
                    v += ve
                    b += be
            
            r = r / (sum_values_kernel-bi) #pour un noyeau a 5, bi = 9; 7, bi = 13; 9, bi = 17; 11, bi = 21
            v = v / (sum_values_kernel-bi)
            b = b / (sum_values_kernel-bi)

            img_convolve[pixelLine][pixel][0] = r
            img_convolve[pixelLine][pixel][1] = v
            img_convolve[pixelLine][pixel][2] = b

    return img_convolve

main()