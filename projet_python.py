import os
from enum import Enum

import matplotlib.image as mplimg
import numpy as np

import functions


# Enumère les différents type d'extension d'image
class ImageExtension(Enum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"


# TODO: pouvoir choisir entre base_test & base_validation
base_path = "res" + os.path.sep + "base_test" + os.path.sep
# récupère les valeurs de ImageExtension
valide_extension = [item.value for item in ImageExtension]

# Charge les images des pieces modele
model_path = "res" + os.path.sep + "piece_model" + os.path.sep
img_2e = (mplimg.imread(model_path + "2e.jpg").copy()).astype(np.uint8)
functions.show_img(img_2e, "Image 2e Original")  # [LOG]
modele_2e = functions.piece_recognition(img_2e)
functions.show_img(modele_2e, "Model 2e après reconnaissance")  # [LOG]

# Compteur pour calculer le taux de reussite de l'algo
traitement_reussite = 0
nb_image_total = 0

# Détection pour chaque fichier image de la base selectionnée
for file in os.listdir(base_path):
    filename = file.split('.', 1)
    file_extension = filename[1]
    # Si le fichier n'est pas une image, on ne le traite pas
    if file_extension not in valide_extension:
        continue
    if filename[0] != "11":
        continue
    print("Traitement : " + file)  # [LOG]
    nb_image_total += 1

    # Récupérer les json contenant les annotations
    json_util_data = {'pieces': [], 'autres': []}  # p-etre mettre plutot un construteur
    json_path = base_path + filename[0] + ".json"
    if os.path.exists(json_path):
        json_util_data = functions.load_jsonfile(json_path)
    else:
        print(f"Fichier non json trouvé : {json_path}")

    # Récup l'image original
    multiplicateur = 1
    # Pour que les pixels des images .png soient des entiers, le multiplacateur doit être égal à 255
    if file_extension == ImageExtension.PNG.value:
        multiplicateur = 255
    img_original = (mplimg.imread(base_path + file).copy() * multiplicateur).astype(np.uint8)
    functions.show_img(img_original, "Image Original")  # [LOG]

    # Récup l'image de validation
    img_validation = functions.create_validation_image(img_original, json_util_data)
    functions.show_img(img_validation, "Image validation")  # [LOG]

    # On réduit la taille des images
    img_valid_resize = functions.image_resize(img_validation, height=800)
    img_resize = functions.image_resize(img_original, height=800)

    """ Détection des pièces """
    # Attention "nb_pieces_trouve" peut être différent de "coord_find_cercle" s'il y a un trop grand nombre de pièce
    img_result, cercles_coords, nb_pieces_trouve = functions.detection_de_pieces(img_resize)
    functions.show_img(img_result, file + " après détection")  # [LOG]

    """ Reconnaitre le côté de la piece et de sa valeur """
    # TODO: à compléter
    # Pour chaque cercle detecté, on l'extrait de l'image resize
    for img_piece in functions.cut_image_into_smaller_pieces(img_resize, cercles_coords):
        img_result_recognition = functions.piece_recognition(img_piece)
        functions.show_img(img_result_recognition, file + " après reconnaissance")  # [LOG]
        img_xor = functions.apply_xor(img_result_recognition, modele_2e)
        functions.show_img(img_xor, file + " après xor")  # [LOG]
        exit(0)  # TODO: arret temporaire pour tester la 1ère piece

    """ Calcul du résultat de la détection des pièces """
    nb_pieces_reelles = len(json_util_data['pieces'])
    # Cherche si chaque pièce detecté est une vrai pièce
    nb_fausse_piece = 0
    if nb_pieces_trouve != 0:
        nb_fausse_piece = functions.calcul_nb_fausse_piece(cercles_coords, img_valid_resize)
    if nb_pieces_trouve == nb_pieces_reelles and nb_fausse_piece == 0:
        traitement_reussite += 1

    pourcentage = "None" if nb_pieces_reelles == 0 else round(nb_pieces_trouve / nb_pieces_reelles * 100)
    print(f"Traitement {file} : Nombre de pièce(s) détectée(s) : " +
          f"{nb_pieces_trouve} sur {nb_pieces_reelles} ({pourcentage}%)"
          f" avec {nb_fausse_piece} faux positive.")

    # break  # TODO: break temporaire pour tester la 1ère image

# Calcule des résultats quantitatifs
pourcentage_final = round(traitement_reussite / nb_image_total * 100)
print(f"Résulat des analyses dans {base_path} : {traitement_reussite} / {nb_image_total}")
print("Fin de la détection !")
