import os
import json
import os.path
import sys

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


def historigramme(img):
    histogram = np.zeros(256, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i][j]] += 1

    return histogram


def histo_cumule(histo):
    """
    Retourne l'historigramme cummulé à partir d'un historigramme
    :param histo un historigramme
    :returns l'historigramme cummulé, le cumulé total des valeurs de l'histo
    """
    h_cumul = np.zeros(len(histo))
    cumul = 0
    for i in range(len(histo)):
        cumul += histo[i]
        h_cumul[i] = cumul
    return h_cumul, cumul


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


def seuillage(img, seuil):
    img_seuil = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] >= seuil:
                img_seuil[i, j] = 255
    return img_seuil


def otsu(img):
    """ Seuillage automatique : Méthode Otsu (version du professeur Sylvain Lobry) """
    meilleur_seuil = 0
    minimun = 10_000_000_000
    histogram = historigramme(img)

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

    # print(f"Le meilleur seuil est {meilleur_seuil}")  # [LOG]
    return meilleur_seuil


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


def dilatation(img, ksize=3):
    elem_struct = np.ones((ksize, ksize), np.uint8)
    img_dilate = cv2.dilate(img, elem_struct, iterations=3)
    return img_dilate


def erosion(img):
    elem_struct = np.ones((3, 3), np.uint8)
    img_dilate = cv2.erode(img, elem_struct, iterations=1)
    return img_dilate


def ouverture(img, size=3):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))


def filtre_sobel(img):
    img_sobel = cv2.Sobel(img, ddepth=-1, dx=1, dy=1, ksize=1)
    return img_sobel


def algo_canny(img):
    meilleur_seuil = otsu(img)
    img_contour = cv2.Canny(img, 50, meilleur_seuil)
    return img_contour


def apply_hough(img_input):
    """ Applique Hough Cercle """
    rows = img_input.shape[0]
    coords_cercles = cv2.HoughCircles(img_input, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                      param1=100, param2=30,
                                      minRadius=1, maxRadius=1000)
    img_hough = None
    if coords_cercles is not None:
        coords_cercles = np.uint16(np.around(coords_cercles))  # ne pas changer le uint16
        for i in coords_cercles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img_input, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            img_hough = cv2.circle(img_input, center, radius, (255, 0, 0), 3)
    else:
        # Si 0 cercle detecté, alors on retourne l'img original + une liste vide de coords de cercle
        img_hough = img_input
        coords_cercles = [[]]

    return coords_cercles, img_hough


def apply_xor(img, other_img):
    img_xor = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != other_img[i][j]:
                img_xor[i][j] = 255
    return img_xor


def compter_pieces(img):
    (contours_piece, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours_piece), contours_piece


def detection_de_pieces(img):
    # TODO 1. Convertir l'image en gris
    img_gris = conversion_en_gris(img)
    # show_img(img_gris, "Gris")  # [LOG]

    # TODO 2. Si historigramme de l'img est trop sombre ou trop clair, on égalise
    # L'égalisation n'aide pas vraiment :/
    # img_histo = historigramme(img_gris)
    # img_egalise = egaliser(img_gris, img_histo)
    # show_img(img_egalise, "Egalisé")  # [LOG]

    # TODO 3. Réduire les bruits avec un lissage de l'image
    # 3.a. Filtre Moyenneur
    # img_lisse = filtre_moyenneur(img_egalise)
    # 3.b. Filtre Median
    img_lisse = filtre_median(img_gris, 9)
    # show_img(img_lisse, "Lissage avec filtre median")  # [LOG]
    # 3.c. Filtre Gaussian
    # img_lisse = filtre_gaussian(img_gris)

    # TODO 4. Trouver le seuil adéquat et l'appliquer
    # meilleur_seuil = otsu(img_lisse)
    # img_binaire = seuillage(img_lisse, meilleur_seuil)
    # ancien code
    """# TODO 5. Detection de contour
    # 5.a Filtre de Sobel
    # img_result = filtre_sobel(img_dilate)
    # 5.b Algo de Canny (le 3. et 4. est compris dedans)"""
    # img_canny = algo_canny(img_lisse)
    # show_img(img_canny, "Canny")  # [LOG]

    # TODO 5.5 HOUGH_CIRCLE
    # Hough circle inclu déjà l'algo de canny
    img_ouvert = ouverture(img_lisse, 15)
    coords_cercles, img_hough = apply_hough(img_ouvert)
    # show_img(img_hough, "HOUGH")  # [LOG]
    img_result = img_hough

    # TODO 6. Si les contours ne sont pas assez gros, le dilater
    # img_dilate = dilatation(img_hough, 3)
    # show_img(img_dilate, "Dilaté")  # [LOG]
    # # cv2.imshow("dilat", img_dilate)
    # img_result = img_dilate

    # TODO 7. Compter les pieces
    # nb_pieces, contours = compter_pieces(img_result)
    #
    # # Affichage des contours détectées  # [LOG]
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # # cv2.imshow("Result", img)
    # # cv2.waitKey(0)

    nb_circle = len(coords_cercles[0])
    if nb_circle > 10:
        nb_circle = 0
    return img_result, coords_cercles[0], nb_circle


def reconnaissance_de_valeur(img, cercles_coords):
    # TODO 1. Réduire les bruits avec un filtre median
    img_lisse = filtre_median(img, ksize=3)
    # TODO 2. Convertir l'imge RGB en HSV
    img_hsv = cv2.cvtColor(img_lisse, cv2.COLOR_RGB2HSV)
    # TODO 3. Recup la couleurs
    img_filtree_orange = detect_colour(img_hsv, [15, 50, 20], [30, 255, 255])
    cv2.imshow("Mask orange", img_filtree_orange)  # [LOG]
    cv2.waitKey(0)
    img_filtree_rouge = detect_colour(img_hsv, [0, 50, 20], [14, 255, 255])
    cv2.imshow("Mask rouge", img_filtree_rouge)  # [LOG]
    cv2.waitKey(0)
    liste_valeurs = []
    for one_piece in cercles_coords:
        piece_value = -1
        pourcentage_orange = get_white_px_pourcentage_in_cercle(one_piece, img_filtree_orange)
        print(one_piece)
        print(pourcentage_orange)
        # le pourcentage de pixel orange sur la pièce est élévée (80%), alors c'est surement :
        if pourcentage_orange >= 80:  # 50, 20 ou 10 centime(s)
            piece_value = 0.5
        elif pourcentage_orange <= 15:  # si c'est inférieur à 15%
            # On regarde le pourcentage de pixel rouge sur la pièce
            pourcentage_rouge = get_white_px_pourcentage_in_cercle(one_piece, img_filtree_rouge)
            if pourcentage_rouge >= 80:  # 5, 2 ou 1 centime(s)
                piece_value = 0.05
        else:
            # On regarde le pourcentage de pixel orange sur la pièce dont la taille est reduite de 50%
            half_piece = [one_piece[0], one_piece[1], one_piece[2] // 2]
            pourcentage_half_orange = get_white_px_pourcentage_in_cercle(half_piece, img_filtree_orange)
            # Si le pourcentage reste haute, alors c une pièce de 2e
            if pourcentage_half_orange >= 80:
                piece_value = 2
            else:  # sinon c'est une pièce de 1 euro
                piece_value = 1
        print("piece_value :", piece_value)
        liste_valeurs.append({"coord": one_piece, "value": piece_value})
    return liste_valeurs


def detect_colour(img_hsv, lowrange, highrange):
    hsv_color1 = np.asarray(lowrange)
    hsv_color2 = np.asarray(highrange)
    mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)
    return mask


def cut_image_into_smaller_pieces(img, list_coords_pieces):
    array_mini_images = []
    for piece in list_coords_pieces:
        # crop l'image originale avec les coordonnées des cerles detectés
        if piece[1] < piece[2] or piece[0] < piece[2]:
            continue
        m_i = (piece[1] - piece[2], piece[0] - piece[2], piece[2] * 2)
        array_mini_images.append(img[(m_i[0]):(m_i[0] + m_i[2]), (m_i[1]):(m_i[1] + m_i[2])])
    return array_mini_images


def show_img(img, img_title):
    plt.figure()
    plt.title(img_title)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def get_white_px_pourcentage_in_cercle(piece_coord, img):
    """
    Retourne le nombre de pixel blanc et de pixel total de la pièce donnée en param.
    ---
    :param piece_coord les coordonnées de la pièce -> (x, y, rayon)
    :param img l'image contenant la pièce
    :returns le pourcentage de nb de px blanc dans le cercle
    """
    white_px = 0
    total_px = 0
    centre = [piece_coord[1], piece_coord[0]]
    rayon = piece_coord[2]
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if np.sqrt((centre[0] - x)**2 + (centre[1] - y)**2) <= rayon:
                total_px += 1
                if round(img[x, y]) == 255:
                    white_px += 1
    if total_px != 0:
        return (white_px / total_px) * 100
    return 0


def load_jsonfile(json_path):
    debut_label_piece = "piece de "
    file = open(json_path)
    data = json.load(file)
    liste_pieces = []  # liste contenant les infos sur chaque pièces de monnaie de l'image
    liste_autres = []  # les choses qui ne sont pas des pièces de monnaie
    for shape in data["shapes"]:
        if shape["label"].startswith(debut_label_piece):
            liste_pieces.append({"label": shape["label"], "points": shape["points"], "shape_type": shape["shape_type"]})
        else:
            liste_autres.append({"label": shape["label"], "points": shape["points"]})
    util_data = {'pieces': liste_pieces, 'autres': liste_autres}
    return util_data


def create_validation_image(img, json_data):
    img_copy = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for piece in json_data["pieces"]:
        if piece["shape_type"] == "circle":
            brut_center = piece["points"][0]
            brut_cercle_pt = piece["points"][1]

            center = (int(brut_center[0]), int(brut_center[1]))
            cercle_pt = (int(brut_cercle_pt[0]), int(brut_cercle_pt[1]))

            rayon = int(np.sqrt((center[0] - cercle_pt[0])**2 + (center[1] - cercle_pt[1])**2))
            cv2.circle(img_copy, center, rayon, (255, 255, 255), -10)
        elif piece["shape_type"] == "polygon":
            list_pts = []
            for points in piece["points"]:
                list_pts.append([int(points[0]), int(points[1])])
            pts = np.array([list_pts], np.int32)  # ne pas changer le int32
            cv2.fillPoly(img_copy, [pts], 255)

    return img_copy


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Fonction pour redimension d'une image sans perdre son allure d'origine
    Source : https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    """
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None, calculate the ratio of the width and construct the dimensions
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def calcul_nb_fausse_piece(cercles_coords, img_valid_resize):
    nb_fausse_piece = 0

    for une_piece in cercles_coords:
        pourcentage = get_white_px_pourcentage_in_cercle(une_piece, img_valid_resize)
        if round(pourcentage) < 70:
            nb_fausse_piece += 1

    return nb_fausse_piece


def enter_images_path():
    arg = ""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    while True:
        print("Choisissez le nom du dossier à traiter :"
              "\n(1) Base d'apprentissage"
              "\n(2) Base de validation"
              "\n(3) Une autre base"
              "\n>>> ", end='')
        if len(arg) > 0:
            base_choice = sys.argv[1]
            arg = ""  # utilise une seul fois
        else:
            base_choice = input()
        if base_choice == "1":
            return "res" + os.path.sep + "base_test" + os.path.sep
        elif base_choice == "2":
            return "res" + os.path.sep + "base_validation" + os.path.sep
        elif base_choice == "3":
            print("Entrer le chemin vers le dossier contenant les images à traiter :"
                  "\n>>> ", end='')
            path = input()
            path += os.path.sep
            if os.path.exists(path):
                return path
            print(f"Le chemin '{path}' n'a pas été trouvé ! Peut-être essayer un chemin relatif.\n")
        else:
            print(f"'{base_choice}' n'est pas un choix invalide !\n")


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
