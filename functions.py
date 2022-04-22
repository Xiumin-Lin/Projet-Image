import json
import os.path
import matplotlib.pyplot as plt
import cv2
import numpy as np


def conversion_en_gris(img):
    # print("DO Conversion en gris")  # [LOG]
    # img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gris = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # img value convert to int prevent the error : "overflow encountered in ubyte_scalars"
            gray_value = int((int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2])) / 3)
            for k in range(3):
                img_gris[i, j] = gray_value

    return img_gris


def historigramme(img):
    # print("DO Historigramme")  # [LOG]
    histogram = np.zeros(256, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i][j]] += 1

    return histogram


def histo_cumule(histo):
    # print("DO Historigramme cumulée")  # [LOG]
    h_cumul = np.zeros(len(histo))
    cumul = 0
    for i in range(len(histo)):
        cumul += histo[i]
        h_cumul[i] = cumul
    return h_cumul, cumul


# Egaliser
def egaliser(img, historigrame):
    # print("DO Egaliser")  # [LOG]
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


# Convolution avec un noyau à 2 dimensions
def convolution_2d(img, noyau):
    # print("DO Convolution 2D")  # [LOG]
    img_convolve_2d = cv2.filter2D(img, ddepth=-1, kernel=noyau)
    return img_convolve_2d


def filtre_moyenneur(img):
    # print("DO Filtre_moyenneur")  # [LOG]
    moy_kernel_5x5 = np.ones((5, 5), np.uint8) / 25
    img_moyenneur = convolution_2d(img, moy_kernel_5x5)
    return img_moyenneur


def filtre_median(img, size=3):
    # TODO
    # print("DO Filtre_median")  # [LOG]
    img_median = cv2.medianBlur(img, ksize=size)
    return img_median


def filtre_gaussian(img, ksize=17):
    # print("DO Filtre_gaussian")  # [LOG]
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
    # TODO
    # print("DO Dilatation")  # [LOG]
    elem_struct = np.ones((ksize, ksize), np.uint8)
    img_dilate = cv2.dilate(img, elem_struct, iterations=3)
    return img_dilate


def erosion(img):
    # TODO
    # print("DO Erosion")  # [LOG]
    elem_struct = np.ones((3, 3), np.uint8)
    img_dilate = cv2.erode(img, elem_struct, iterations=1)
    return img_dilate


# Seuillage automatique : Méthode Otsu (version du prof)
def otsu(img):
    # print("Use otsu : ")  # [LOG]
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


def seuillage(img, seuil):
    # print("DO Seuillage")  # [LOG]
    img_seuil = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] >= seuil:
                img_seuil[i, j] = 1
    return img_seuil


def filtre_sobel(img):
    # TODO
    # print("DO Filtre_sobel")  # [LOG]
    img_sobel = cv2.Sobel(img, ddepth=-1, dx=1, dy=1, ksize=1)
    return img_sobel


def algo_canny(img):
    # TODO
    # print("DO Algo_canny")  # [LOG]
    meilleur_seuil = otsu(img)
    img_contour = cv2.Canny(img, 50, meilleur_seuil)
    return img_contour


def compter_pieces(img):
    # TODO : Faire notre propre detection de cercle
    # print("DO Compter_pieces")  # [LOG]
    (contours_piece, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours_piece), contours_piece


def detection_de_pieces(img):
    # TODO 1. Convertir l'image en gris
    img_gris = conversion_en_gris(img)
    # show_img(img_gris, "Gris")  # [LOG]
    # show_img(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "Gris by cv2")  # [LOG]

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


def piece_recognition(img_piece):
    # TODO (Part 2) 1. Resize
    img_piece_resize = image_resize(img_piece, height=200)

    # TODO (Part 2) 2. Convertir l'image en gris
    # Reminder : On pourrait convertir direct l'img en gris avant de les passer en param
    img_gris = conversion_en_gris(img_piece_resize)
    show_img(img_gris, "Gris")  # [LOG]

    # TODO (Part 2) 3. Si historigramme de l'img est trop sombre ou trop clair, on égalise
    # img_histo = historigramme(img_gris)
    # img_egalise = egaliser(img_gris, img_histo)
    # show_img(img_egalise, "Egalisé")  # [LOG]

    # TODO (Part 2) 4. Réduire les bruits avec un lissage de l'image
    # 3.a. Filtre Moyenneur
    # img_lisse = filtre_moyenneur(img_egalise)
    # 3.b. Filtre Median
    # img_lisse = filtre_median(img_gris, 9)
    # 3.c. Filtre Gaussian
    img_lisse = filtre_gaussian(img_gris, ksize=3)
    show_img(img_lisse, "Lissage avec filtre gaussian")  # [LOG]

    # TODO (Part 2) 5. Detection de contour (Algo de Canny)
    img_canny = algo_canny(img_lisse)
    show_img(img_canny, "Canny")  # [LOG]

    # TODO (Part 2) 6.1 Extraire le chiffre (comme ce qu'on a fait avec la piece)
    # TODO (Part 2) 6.1.1. Faire un XOR avec le chiffre modele (1, 2 et 5)

    # TODO (Part 2) 6.2 Si pas de chiffre alors c face

    img_result = img_canny
    return img_result


def apply_xor(img, other_img):
    img_xor = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != other_img[i][j]:
                img_xor[i][j] = 255
    return img_xor


def ouverture(img, size=3):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))


def cut_image_into_smaller_pieces(img, coord_array):
    array_mini_images = []
    for i in coord_array:
        # crop l'image originale avec les coordonnées des cerles detectés
        m_i = (i[1] - i[2], i[0] - i[2], i[2] * 2)
        array_mini_images.append(img[(m_i[0]):(m_i[0] + m_i[2]), (m_i[1]):(m_i[1] + m_i[2])])
    return array_mini_images


def apply_hough(img_input):
    rows = img_input.shape[0]
    coords_cercles = cv2.HoughCircles(img_input, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                      param1=100, param2=30,
                                      minRadius=1, maxRadius=1000)
    img_hough = None
    if coords_cercles is not None:
        coords_cercles = np.uint16(np.around(coords_cercles))
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


def show_img(img, img_title):
    plt.figure()
    plt.title(img_title)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


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
    print(img.shape)
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
            pts = np.array([list_pts], np.int32)
            cv2.fillPoly(img_copy, [pts], 255)

    return img_copy


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


def calcul_nb_fausse_piece(cercles_coords, img_valid_resize):
    nb_fausse_piece = 0

    for piece in cercles_coords:
        print(piece)
        white_px = 0
        total_px = 0
        centre = [piece[1], piece[0]]
        rayon = piece[2]
        for x in range(img_valid_resize.shape[0]):
            for y in range(img_valid_resize.shape[1]):
                if np.sqrt((centre[0] - x)**2 + (centre[1] - y)**2) <= rayon:
                    total_px += 1
                    if round(img_valid_resize[x, y]) == 255:
                        white_px += 1
        if round(white_px / total_px * 100) < 70:
            nb_fausse_piece += 1

    return nb_fausse_piece
