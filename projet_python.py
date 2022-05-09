import os
from enum import Enum

import matplotlib.image as mplimg
import numpy as np

import functions
import traitement_util.util_misc as tutil


class ImageExtension(Enum):
    """ Enumère les différents type d'extension d'image """
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"


# Récupère les valeurs de ImageExtension
valide_extension = [item.value for item in ImageExtension]
# Les images à traiter sont redimensionnées à la hauteur en pixel suivante.
IMG_HEIGHT = 800
# Le chemin menant au dossier contenant les images à traiter
BASE_PATH = functions.enter_images_path()

""" Compteur pour calculer le taux de reussite de l'algo """
detection_reussite = 0
reconnaissance_reussite = 0
nb_image_total = 0

print(f"----- [ Début de l'analyse du dossier : {BASE_PATH} ] -----", end="\n\n")
""" Détection pour chaque fichier image de la base selectionnée """
for file in os.listdir(BASE_PATH):
    filename = file.split('.', 1)
    if len(filename) == 1:
        continue
    file_extension = filename[1]
    """ Si le fichier n'est pas une image, on ne le traite pas """
    if file_extension not in valide_extension:
        continue
    if filename[0] != "11":
        continue
    print(f"Traitement : {file} ", end='')  # [LOG]
    nb_image_total += 1

    """ Récupére le json contenant les annotations """
    json_util_data = {'pieces': [], 'autres': []}  # p-etre mettre plutot un construteur
    json_path = BASE_PATH + filename[0] + ".json"
    if os.path.exists(json_path):
        json_util_data = functions.load_jsonfile(json_path)
    else:
        print(f"(Fichier non json trouvé : {json_path}) ", end='')

    """ Récup l'image originale
    Pour que les pixels des images .png soient des entiers, le multiplacateur doit être égal à 255 """
    multiplicateur = 1
    if file_extension == ImageExtension.PNG.value:
        multiplicateur = 255
    img_originale = (mplimg.imread(BASE_PATH + file).copy() * multiplicateur).astype(np.uint8)
    img_resize = tutil.image_resize(img_originale, height=IMG_HEIGHT)  # On réduit la taille l'image
    # functions.show_img(img_resize, "Image originale redimensionnée")  # [LOG]

    """ Détection des pièces """
    # Attention "nb_pieces_trouvees" peut être différent de "coord_find_cercle" s'il y a un trop grand nombre de pièce
    img_result, cercles_coords, nb_pieces_trouvees = functions.detection_de_pieces(img_resize)
    # functions.show_img(img_result, file + " après détection")  # [LOG]

    """ Reconnaitre la valeur de la piece """
    liste_pieces_detectees = functions.reconnaissance_de_valeur(img_resize, cercles_coords)

    """ Calcul du résultat de la détection des pièces """
    nb_pieces_reelles = len(json_util_data['pieces'])
    # Cherche si chaque pièce detectée est une vrai pièce
    nb_fausse_piece = 0
    dico_bonne_p_detectee = {"1e": 0, "2e": 0, "centimes": 0, "petits_centimes": 0, "unknown": 0}
    liste_mauvaise_p_detectee = []
    if nb_pieces_trouvees != 0:
        # Récup l'image de validation
        img_validation = functions.create_validation_image(img_originale, json_util_data)
        # functions.show_img(img_validation, "Image validation")  # [LOG]
        # On réduit la taille de l'image
        img_valid_resize = tutil.image_resize(img_validation, height=IMG_HEIGHT)
        # On calcule et recupere les erreurs d'analyse des images
        nb_fausse_piece, dico_bonne_p_detectee, liste_mauvaise_p_detectee = functions.calcul_erreur_analyse(
            liste_pieces_detectees, img_valid_resize)
        # On retire les pieces trouvées qui ne sont pas des pièces
        nb_pieces_trouvees -= dico_bonne_p_detectee["unknown"]
    if nb_pieces_trouvees == nb_pieces_reelles and nb_fausse_piece == 0:
        if len(liste_mauvaise_p_detectee) == 0:
            reconnaissance_reussite += 1
        detection_reussite += 1

    """ Affichage les résultats de la détection pour une image """
    functions.show_piece_analyse_result(nb_pieces_trouvees, nb_pieces_reelles, nb_fausse_piece,
                                        dico_bonne_p_detectee, liste_mauvaise_p_detectee)

# Calcule des résultats quantitatifs finals
print("\n----- [ Fin de la détection ! ] -----")
if nb_image_total != 0:
    pourcentage_final = round(detection_reussite / nb_image_total * 100)
    print(f"\nRésulat des analyses dans {BASE_PATH} :"
          f"\n\tDétection : {detection_reussite} / {nb_image_total}"
          f"\n\tReconnaissance : {reconnaissance_reussite} / {nb_image_total}")
else:
    print("Aucun fichier image valide n'a été trouvé !")
print("\n----- [ Fin de la détection ! ] -----")
