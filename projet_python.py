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

# Détection pour chaque fichier image de la base selectionnée
for filename in os.listdir(base_path):
    file_extension = filename.split('.')[1]
    # Si le fichier n'est pas une image, on ne le traite pas
    if file_extension not in valide_extension:
        continue

    print("Traitement : " + filename)
    multiplicateur = 1
    # Pour que les pixels des images .png soient des entiers, le multiplacateur doit être égal à 255
    if file_extension == ImageExtension.PNG.value:
        multiplicateur = 255
    img = (mplimg.imread(base_path + filename).copy() * multiplicateur).astype(np.uint8)

    # Affichage
    plt.figure()
    plt.title(f"{filename} originale")
    plt.imshow(img)
    plt.show()

    # Detection des pièces
    img_result, nb_pieces = functions.detection_de_pieces(img)
    print(f"Pièce(s) détectée(s) : {nb_pieces}")

    plt.figure()
    plt.title(f"{filename} après détection")
    plt.imshow(img_result)
    plt.show()

    # Interprétation du côté de la piece et de sa valeur
    # TODO: à compléter

    break  # TODO: break temporaire pour tester la 1ère image

print("Fin de la détection !")
