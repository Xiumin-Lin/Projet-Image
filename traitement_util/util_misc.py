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


def egalisation(img, histo):
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


def algo_canny(img):
    meilleur_seuil = otsu(img)
    img_contour = cv2.Canny(img, 50, meilleur_seuil)
    return img_contour


def compter_pieces(img):
    (contours_piece, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours_piece), contours_piece


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


def detect_colour(img_hsv, lowrange, highrange):
    low_hsv_color = np.asarray(lowrange)
    high_hsv_color = np.asarray(highrange)
    mask = cv2.inRange(img_hsv, low_hsv_color, high_hsv_color)
    return mask


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


def cut_image_into_smaller_pieces(img, list_coords_pieces):
    array_mini_images = []
    for piece in list_coords_pieces:
        # crop l'image originale avec les coordonnées des cerles detectés
        if piece[1] < piece[2] or piece[0] < piece[2]:
            continue
        m_i = (piece[1] - piece[2], piece[0] - piece[2], piece[2] * 2)
        array_mini_images.append(img[(m_i[0]):(m_i[0] + m_i[2]), (m_i[1]):(m_i[1] + m_i[2])])
    return array_mini_images


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Fonction pour redimensionner une image sans perdre son allure d'origine
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
