import os
from enum import Enum

import matplotlib.image as mplimg
import matplotlib.pyplot as plt
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
    """if filename[0] != "18":
        continue"""
    print("Traitement : " + file)  # [LOG]
    nb_image_total += 1

    # Récupérer les json contenant les annotations
    json_util_data = {'pieces': [], 'autres': []}  # p-etre mettre plutot un construteur
    json_path = base_path + filename[0] + ".json"
    if os.path.exists(json_path):
        json_util_data = functions.load_jsonfile(json_path)
    else:
        print(f"Fichier non json trouvé : {json_path}")

    multiplicateur = 1
    # Pour que les pixels des images .png soient des entiers, le multiplacateur doit être égal à 255
    if file_extension == ImageExtension.PNG.value:
        multiplicateur = 255
    img = (mplimg.imread(base_path + file).copy() * multiplicateur).astype(np.uint8)
    functions.show_img(img, "Image Original")  # [LOG]

    img_validation = functions.create_validation_image(img, json_util_data)
    functions.show_img(img_validation, "Image validation")  # [LOG]
    img_resize = functions.image_resize(img, height=800)

    # Détection des pièces
    img_result, nb_pieces_trouve = functions.detection_de_pieces(img_resize)
    # TODO: A compléter avec la reconnaissance des faces
    nb_pieces_reelles = len(json_util_data['pieces'])
    if nb_pieces_trouve == nb_pieces_reelles:
        traitement_reussite += 1

    pourcentage = "None" if nb_pieces_reelles == 0 else nb_pieces_trouve / nb_pieces_reelles * 100
    print(f"Traitement {file} : Nombre de pièce(s) détectée(s) : " +
          f"{nb_pieces_trouve} sur {nb_pieces_reelles} ({pourcentage}%)")

    functions.show_img(img_result, file + " après détection")  # [LOG]
    # Interprétation du côté de la piece et de sa valeur
    # TODO: à compléter

    # break  # TODO: break temporaire pour tester la 1ère image

pourcentage_final = traitement_reussite / nb_image_total * 100
print(f"Résulat des analyses dans {base_path} : {traitement_reussite} / {nb_image_total}")
print("Fin de la détection !")
