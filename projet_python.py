import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
from enum import Enum
import functions


# TODO a complter
class Filtre(Enum):
    GAUSSIAN = [0, 0, 0]


# Saisie du nom du fichier
# if len(sys.argv) < 2:
#     print("utilisez le program avec l'option '-help' pour un detail d'utilisation")
#     exit()

# prefix = sys.argv[1]
for i in range(10):
    # i represente le num de l'img que vous souhaitez tester
    filename, multiplicateur = functions.get_file(i)

    # recupere l'img
    print(filename)
    img = (mplimg.imread("res/" + filename).copy()
           * multiplicateur).astype(np.uint8)
    # TODO detecter si trop sombre ou trop
    img_egalised = functions.egaliser(img)
    # TODO reduire les bruits (blur)
    img_blured = functions.convolution(img_egalised, Filtre.GAUSSIAN)
    # TODO appliquer le seuil
    # best_seuil = functions.otsu(img_blured)
    # img_seuil = functions.seuil(img, best_seuil)

    plt.figure()
    plt.title(filename)
    # plt.imshow(img_seuil, cmap=plt.cm.gray)
    plt.imshow(img_blured, cmap=plt.cm.gray)
    plt.show()
