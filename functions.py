import os.path

import cv2
import numpy as np


def conversion_en_gris(img):
    # TODO
    print("Do Conversion en gris")
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gris


def historigramme(img):
    print("DO Historigramme")
    histogram = np.zeros(256, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i][j]] += 1

    return histogram


def histo_cumule(histo):
    print("DO Historigramme cumulée")
    h_cumul = np.zeros(len(histo))
    cumul = 0
    for i in range(len(histo)):
        cumul += histo[i]
        h_cumul[i] = cumul
    return h_cumul, cumul


# Egaliser
def egaliser(img, historigrame):
    print("DO Egaliser")
    h_cumule, cumul = histo_cumule(historigrame)
    img_egalise = np.zeros(img.shape, dtype=np.uint8)
    n = len(historigrame)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_value = ((n / cumul) * h_cumule[img[i][j]]) - 1
            # img_egalise = max(0, newValue)
            if new_value > 0:
                img_egalise[i][j] = new_value
            else:
                img_egalise[i][j] = 0
    return img_egalise


# Convolution avec un noyau à 1 dimension
def convolution_1d(img, noyau):
    # TODO
    print("DO Convolution 1D")
    convolve_1d = img
    return convolve_1d


# Convolution avec un noyau à 2 dimensions
def convolution_2d(img, noyau):
    # TODO
    print("DO Convolution 2D")
    convolve_2d = img
    return convolve_2d


def filtre_moyenneur(img):
    # TODO
    print("DO Filtre_moyenneur")
    img_moyenneur = img
    return img_moyenneur


def filtre_median(img):
    # TODO
    print("DO Filtre_median")
    img_median = img
    return img_median


def filtre_gaussian(img):
    # TODO
    print("DO Filtre_gaussian")
    return cv2.GaussianBlur(img, (17, 17), 0)


def dilatation(img):
    # TODO
    print("DO Dilatation")
    noyau = np.ones((5, 5), np.uint8)
    img_dilate = cv2.dilate(img, noyau, iterations=2)
    return img_dilate


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
    meilleur_seuil = otsu(img)
    img_contour = cv2.Canny(img, 30, meilleur_seuil)
    return img_contour


def compter_pieces(img):
    # TODO : Faire notre propre detection de cercle
    print("DO Compter_pieces")
    (contours_piece, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours_piece), contours_piece


def detection_de_pieces(img):
    # TODO 1. Convertir l'image en gris
    img_gris = conversion_en_gris(img)

    # plt.figure()
    # plt.title("Gris")
    # plt.imshow(img_gris, cmap=plt.cm.gray)
    # plt.show()

    # TODO 2. Si historigramme de l'img est trop sombre ou trop clair, on égalise
    # L'égalisation n'aide pas vraiment :/
    # img_histo = historigramme(img_gris)
    # img_egalise = egaliser(img_gris, img_histo)

    # plt.figure()
    # plt.title("Egalisé")
    # plt.imshow(img_egalise, cmap=plt.cm.gray)
    # plt.show()

    # TODO 3. Réduire les bruits avec un lissage de l'image
    # 3.a. Filtre Moyenneur
    # img_lisse = filtre_moyenneur(img_egalise)
    # 3.b. Filtre Median
    # img_lisse = filtre_moyenneur(img_egalise)
    # 3.c. Filtre Gaussian
    img_lisse = filtre_gaussian(img_gris)

    # plt.figure()
    # plt.title("Lissage")
    # plt.imshow(img_lisse, cmap=plt.cm.gray)
    # plt.show()

    # TODO 4. Trouver le seuil adéquat et l'appliquer
    # meilleur_seuil = otsu(img_lisse)
    # img_binaire = seuillage(img_lisse, meilleur_seuil)
    # TODO 5. Si les contours ne sont pas assez gros, le dilater
    img_dilate = dilatation(img_lisse)
    #
    # plt.figure()
    # plt.title("Dilate")
    # plt.imshow(img_dilate, cmap=plt.cm.gray)
    # plt.show()

    # TODO 6. Detection de contour
    # 4.a Filtre de Sobel
    # img_result = filtre_sobel(img_dilate)
    # 4.b Filtre de Kirsh
    # img_result = filtre_kirsh(img_dilate)
    # 4.c Algo de canny (le 3. et 4. est compris dedans)
    img_result = algo_canny(img_dilate)

    # plt.figure()
    # plt.title("Canny")
    # plt.imshow(img_result, cmap=plt.cm.gray)
    # plt.show()

    # TODO 7. Compter les pieces
    nb_pieces, contours = compter_pieces(img_result)

    # Affichage des contours détectées
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)

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


# Fonction pour redimension d'une image sans perdre son allure d'origine
# Source : https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
