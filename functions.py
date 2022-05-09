import os
import json
import os.path
import re
import sys
import traitement_util.filtres as fltr
import traitement_util.morphologies as morph
import traitement_util.util_misc as tutil
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_img(img, img_title):
    plt.figure()
    plt.title(img_title)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def algo_canny(img):
    meilleur_seuil = tutil.otsu(img)
    img_contour = cv2.Canny(img, 50, meilleur_seuil)
    return img_contour

def compter_pieces(img):
    (contours_piece, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours_piece), contours_piece

def detection_de_pieces(img):
    # TODO 1. Convertir l'image en gris
    img_gris = fltr.conversion_en_gris(img)
    # show_img(img_gris, "Gris")  # [LOG]

    # TODO 2. Réduire les bruits avec un lissage de l'image
    # Filtre Median
    img_lisse = fltr.filtre_median(img_gris, 9)
    # show_img(img_lisse, "Lissage avec filtre median")  # [LOG]

    # TODO 3 HOUGH_CIRCLE
    # Hough circle inclu déjà l'algo de canny
    img_ouvert = morph.ouverture(img_lisse, 15)
    coords_cercles, img_hough = tutil.apply_hough(img_ouvert)
    # show_img(img_hough, "HOUGH")  # [LOG]
    img_result = img_hough

    nb_circle = len(coords_cercles[0])
    if nb_circle > 10:
        nb_circle = 0
    return img_result, coords_cercles[0], nb_circle

def reconnaissance_de_valeur(img, cercles_coords):
    # TODO 1. Réduire les bruits avec un filtre median
    img_lisse = fltr.filtre_median(img, ksize=3)
    # TODO 2. Convertir l'image RGB en HSV
    img_hsv = cv2.cvtColor(img_lisse, cv2.COLOR_RGB2HSV)
    # TODO 3. Recup l'image contenant qu les couleurs desirée (un orange et un rouge)
    img_filtree_orange = tutil.detect_colour(img_hsv, [15, 50, 20], [30, 255, 255])
    cv2.imshow("Mask orange", img_filtree_orange)  # [LOG]
    cv2.waitKey(0)
    img_filtree_rouge = tutil.detect_colour(img_hsv, [0, 50, 20], [14, 255, 255])
    cv2.imshow("Mask rouge", img_filtree_rouge)  # [LOG]
    cv2.waitKey(0)
    liste_pieces_detectees = []
    # TODO 4. Attribuer la valeur de chaque piece via les img_filtree_orange et img_filtree_rouge obtenus
    for one_piece in cercles_coords:
        print("*", end='')
        piece_value = "unknown"
        pourcentage_orange = tutil.get_white_px_pourcentage_in_cercle(one_piece, img_filtree_orange)
        # le pourcentage de pixel orange sur la pièce est élévée (80%), alors c'est surement :
        if pourcentage_orange >= 80:  # 50, 20 ou 10 centime(s)
            piece_value = 0.5
        elif pourcentage_orange <= 15:  # si c'est inférieur à 15%
            # On regarde le pourcentage de pixel rouge sur la pièce
            pourcentage_rouge = tutil.get_white_px_pourcentage_in_cercle(one_piece, img_filtree_rouge)
            if pourcentage_rouge >= 80:  # 5, 2 ou 1 centime(s)
                piece_value = 0.05
        else:
            # On regarde le pourcentage de pixel orange sur la pièce dont la taille est reduite de 50%
            half_piece = [one_piece[0], one_piece[1], one_piece[2] // 2]
            pourcentage_half_orange = tutil.get_white_px_pourcentage_in_cercle(half_piece, img_filtree_orange)
            # Si le pourcentage reste haut, alors c une pièce de 2e
            if pourcentage_half_orange >= 80:
                piece_value = 2
            else:  # sinon c'est une pièce de 1 euro
                piece_value = 1
        liste_pieces_detectees.append({"coords": one_piece, "value": piece_value})
    print()
    return liste_pieces_detectees

def load_jsonfile(json_path):
    debut_label_piece = "piece de "
    file = open(json_path)
    data = json.load(file)
    liste_pieces = []  # liste contenant les infos sur chaque pièces de monnaie de l'image
    liste_autres = []  # les choses qui ne sont pas des pièces de monnaie
    for shape in data["shapes"]:
        if shape["label"].startswith(debut_label_piece):
            value_str = shape["label"].split(debut_label_piece)[1]
            value = re.split(" ", value_str, 1)[0]
            liste_pieces.append({"label": shape["label"], "value": value, "points": shape["points"], "shape_type": shape["shape_type"]})
        else:
            liste_autres.append({"label": shape["label"], "points": shape["points"]})
    util_data = {'pieces': liste_pieces, 'autres': liste_autres}
    return util_data

def create_validation_image(img, json_data):
    img_copy = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    for piece in json_data["pieces"]:
        color_value = 255
        value_str = re.split("[e c]", piece["value"], 1)
        if piece["value"][-1] == 'e' or int(value_str[0]) > 5:
            color_value -= int(value_str[0])
        else:
            color_value -= int(value_str[0]) * 7  # TODO bidoullage

        if piece["shape_type"] == "circle":
            brut_center = piece["points"][0]
            brut_cercle_pt = piece["points"][1]

            center = (int(brut_center[0]), int(brut_center[1]))
            cercle_pt = (int(brut_cercle_pt[0]), int(brut_cercle_pt[1]))

            rayon = int(np.sqrt((center[0] - cercle_pt[0])**2 + (center[1] - cercle_pt[1])**2))
            cv2.circle(img_copy, center, rayon, color_value, -10)
        elif piece["shape_type"] == "polygon":
            list_pts = []
            for points in piece["points"]:
                list_pts.append([int(points[0]), int(points[1])])
            pts = np.array([list_pts], np.int32)  # ne pas changer le int32
            cv2.fillPoly(img_copy, [pts], color_value)

    return img_copy

def calcul_erreur_analyse(liste_pieces_detectees, img_valid_resize):
    nb_fausse_piece = 0
    dico_bonne_p_detectee = {"1e": 0, "2e": 0, "centimes": 0, "petits_centimes": 0, "unknown": 0}
    liste_mauvaise_p_detectee = []
    # pour chaque pièce detectée
    for p in liste_pieces_detectees:
        # Si la valeur trouvée d'une pièce est unknown, on le note dans le dico et on passe à la pièce suivante
        if p["value"] == "unknown":
            dico_bonne_p_detectee["unknown"] += 1
            continue
        p_coords = p["coords"]
        p_valeur_reelle = img_valid_resize[p_coords[1]][p_coords[0]]
        print(f"Valeur trouvée = {p['value']} vs valeur reelle {p_valeur_reelle}")  # [LOG]
        img_valid_seuil = tutil.seuillage(img_valid_resize, 200)
        pourcentage = tutil.get_white_px_pourcentage_in_cercle(p_coords, img_valid_seuil)
        # Si la piece trouvee ne correspond pas à la piece reelle à 70% près, alors cette piece n'est pas bonne
        if round(pourcentage) < 70:
            nb_fausse_piece += 1
        else:  # Sinon
            #
            valeur_trouvee = p["value"]
            # Si la valeur réelle est divisible par 7 alors on s'attend que la valeur trouvée soit < 10 centimes
            valeur_reelle = 255 - p_valeur_reelle
            if valeur_reelle % 7 == 0:
                if valeur_trouvee < 0.10:
                    dico_bonne_p_detectee["petits_centimes"] += 1
                else:
                    liste_mauvaise_p_detectee.append({"find": valeur_trouvee, "real": valeur_reelle})
            elif valeur_reelle % 10 == 0:
                if 0.50 >= valeur_trouvee >= 0.10:
                    dico_bonne_p_detectee["centimes"] += 1
                else:
                    liste_mauvaise_p_detectee.append({"find": valeur_trouvee, "real": valeur_reelle})
            elif valeur_reelle == valeur_trouvee and valeur_trouvee >= 1:
                dico_bonne_p_detectee[f"{valeur_trouvee}e"] += 1
            else:
                liste_mauvaise_p_detectee.append({"find": valeur_trouvee, "real": valeur_reelle})

    return nb_fausse_piece, dico_bonne_p_detectee, liste_mauvaise_p_detectee

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

def show_piece_analyse_result(nb_pieces_trouvees, nb_pieces_reelles, nb_fausse_piece, dico_bonne_p_detectee):
    total_p_reconnues = 0
    for key in dico_bonne_p_detectee:
        if key != "unknown":
            total_p_reconnues += dico_bonne_p_detectee[key]

    pourcentage = "None" if nb_pieces_reelles == 0 else round(nb_pieces_trouvees / nb_pieces_reelles * 100)
    pourcentage_reconnaissance = "None" if nb_pieces_reelles == 0 else round(total_p_reconnues / nb_pieces_reelles * 100)

    print(f"\n\tNombre de pièce(s) détectée(s) : {nb_pieces_trouvees} sur {nb_pieces_reelles} ({pourcentage}%)" +
          f" avec {nb_fausse_piece} faux positive." +
          f"\n\tNombre de pièce(s) reconnue(s) : {total_p_reconnues} sur {nb_pieces_reelles} ({pourcentage_reconnaissance}%) - (" +
          f"{dico_bonne_p_detectee['1e']} * 1e ; {dico_bonne_p_detectee['2e']} * 2e ; "
          f"{dico_bonne_p_detectee['centimes']} * pieces de [10, 20, 50]c ; "
          f"{dico_bonne_p_detectee['petits_centimes']} * pieces de [1, 2, 5]c)")
    if dico_bonne_p_detectee['unknown'] != 0:
        print(f"\n\tIl y a {dico_bonne_p_detectee['unknown']} pièce(s) unknown non comptabilisé.")



