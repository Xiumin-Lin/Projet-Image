import os.path

import numpy as np


def conversion_en_gris(img):
    # TODO
    print("Do Conversion en gris")
    img_gris = []
    return img_gris


def historigramme(img):
    print("DO Historigramme")
    histogram = np.zeros(256, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i][j]] += 1

    return histogram


def histo_cumule(img):
    # TODO
    print("DO Historigramme cumulée")
    h_cumul = []
    return h_cumul


# Egaliser
def egaliser(img, historigrame):
    # TODO
    print("DO Egaliser")
    return img


# Convolution avec un noyau à 1 dimension
def convolution_1d(img, noyau):
    # TODO
    print("DO Convolution 1D")
    return img


# Convolution avec un noyau à 2 dimensions
def convolution_2d(img, noyau):
    # TODO
    print("DO Convolution 2D")
    return img


def filtre_moyenneur(img):
    # TODO
    print("DO Filtre_moyenneur")
    return img


def filtre_median(img):
    # TODO
    print("DO Filtre_median")
    return img


def filtre_gaussian(img):
    # TODO
    print("DO Filtre_gaussian")
    return img


def dilatation(img):
    # TODO
    print("DO Dilatation")
    return img


# Seuillage automatique : Méthode Otsu (version du prof)
def otsu(img):
    print("Use otsu : ")
    meilleur_seuil = 0
    minimun = 10_000_000_000
    histogram = historigramme(img)

    # Affichage de l'historigramme
    # for i in range(len(histogram)):
    #     print(i, " : ", histogram[i])

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
        w1 /= (img.shape[0] * img.shape[1])  # On calcul la moyenne w1/nbPxTotal
        w2 /= (img.shape[0] * img.shape[1])  # La 2e moyenne

        # Variance
        s1 = 0
        s2 = 0
        for i in range(0, seuil):
            s1 += ((i - mu1)**2) * histogram[i]
        for i in range(seuil, 256):
            s2 += ((i - mu2)**2) * histogram[i]

        intra_class_var = w1 * s1 + w2 * s2
        if intra_class_var < minimun:
            meilleur_seuil = seuil
            minimun = intra_class_var
            # print(f"Nouveau meilleur seuil : {seuil}")

    print(f"Le meilleur seuil est {meilleur_seuil}")
    return meilleur_seuil


def seuillage(img, seuil):
    # TODO
    print("DO Seuillage")
    return img


def filtre_sobel(img):
    # TODO
    print("DO Filtre_sobel")
    return img


def filtre_kirsh(img):
    # TODO
    print("DO Filtre_kirsh")
    return img


def algo_canny(img):
    # TODO
    print("DO Algo_canny")
    # 1. Réduction de bruit (Filtre Gaussian, sa taille est importante)
    # 2. Gradient d'intensité
    # 3. Suppression des non-maxima
    # 4. Seuillage des contours (2 seuils)
    return img


def compter_pieces(img):
    # TODO
    print("DO Compter_pieces")
    nb_piece = 0
    return nb_piece


def detection_de_pieces(img):
    # TODO 1. Convertir l'image en gris
    img_gris = conversion_en_gris(img)
    # TODO 2. Si historigramme de l'img est trop sombre ou trop clair, on égalise
    img_histo = histo_cumule(img_gris)
    img_egalise = egaliser(img, img_histo)
    # TODO 3. Réduire les bruits avec un lissage de l'image
    # 3.a. Filtre Moyenneur
    # img_lisse = filtre_moyenneur(img_egalise)
    # 3.b. Filtre Median
    # img_lisse = filtre_moyenneur(img_egalise)
    # 3.c. Filtre Gaussian
    img_lisse = filtre_gaussian(img_egalise)

    # TODO 4. Trouver le seuil adéquat et l'appliquer
    meilleur_seuil = otsu(img_lisse)
    img_binaire = seuillage(img_lisse, meilleur_seuil)
    # TODO 5. Si les contours ne sont pas assez gros, le dilater
    img_dilate = dilatation(img_binaire)
    # TODO 6. Detection de contour
    # 4.a Filtre de Sobel
    # img_result = filtre_sobel(img_dilate)
    # 4.b Filtre de Kirsh
    # img_result = filtre_kirsh(img_dilate)
    # 4.c Algo de canny (le 3. et 6. est compris dedans)
    img_result = algo_canny(img_dilate)
    # TODO 7. Compter les pieces
    nb_pieces = compter_pieces(img_result)

    return img_result, nb_pieces


def get_file(prefix):
    # multiplicateur = 1 pour afficher une img .jpeg ou jpg dans sa couleur d'origine
    # multiplicateur = 255 pour afficher une img .png dans sa couleur d'origine
    multiplicateur = 1
    filename = ""
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
        print("il n'y a pas de fichier " + str(prefix)
              + " .jpg, .jpeg ou .png, utilisez le program avec '-help' pour un detail d'utilisation")
        exit()
    return filename, multiplicateur
