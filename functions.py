import numpy as np
import sys
import os.path


# Egaliser
def egaliser(img):
    # TODO
    print("DO Egaliser")
    return img


# Convolution
def convolution(img, filtre):
    # TODO
    print("DO convolution")
    return img


# Otsu (version du prof)
def otsu(img):
    print("Use otsu : ")
    meilleur_seuil = 0
    minimun = 10_000_000_000

    histogram = np.zeros(256, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i][j]] += 1

    for i in range(len(histogram)):
        print(i, " : ", histogram[i])

    for seuil in range(256):
        w1 = 0
        w2 = 0
        mu1 = 0
        mu2 = 0
        for i in range(0, seuil):
            w1 += histogram[i]
            mu1 += i * histogram[i]
        for i in range(seuil, 256):
            w2 += histogram[i]
            mu2 += i * histogram[i]

        if w1 == 0 or w2 == 0:
            continue

        mu1 /= w1
        mu2 /= w2
        w1 /= (img.shape[0] * img.shape[1])
        w2 /= (img.shape[0] * img.shape[1])

        # Variance
        s1 = 0
        s2 = 0
        for i in range(0, seuil):
            s1 += ((i - mu1) ** 2) * histogram[i]
        for i in range(seuil, 256):
            s2 += ((i - mu2) ** 2) * histogram[i]

        intra_class_var = w1 * s1 + w2 * s2
        if intra_class_var < minimun:
            meilleur_seuil = seuil
            minimun = intra_class_var
            print(f"Nouveau meilleur seuil : {seuil}")

    print(f"Le meilleur seuil est {meilleur_seuil}")
    return meilleur_seuil


def seuil(img, seuil):
    # TODO
    return img


#
def get_file(prefix):
    # mul = 1 pour afficher une img .jpeg ou jpg dans sa couleur d'origine
    # mul = 255 pour afficher une img .png dans sa couleur d'origine
    filename = ""
    multiplicateur = 1
    file_location = "./res/" + str(prefix)
    if os.path.isfile(file_location + ".jpeg"):
        filename = str(prefix) + ".jpeg"
    elif os.path.isfile(file_location + ".jpg"):
        filename = str(prefix) + ".jpg"
    elif os.path.isfile(file_location + ".png"):
        filename = str(prefix) + ".png"
        multiplicateur = 255
    elif prefix == "-help":
        # TODO
        exit()
    else:
        print("il n'y a pas de fichier " + str(prefix) +
              " .jpg, .jpeg ou .png, utilisez le program avec '-help' pour un detail d'utilisation")
        exit()
    return filename, multiplicateur
