import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import sys
import os.path
import functions

# Saisie du nom du fichier
if len(sys.argv) < 2:
    print("utilisez le program avec l'option '-help' pour un detail d'utilisation")
    exit()

prefix = sys.argv[1]

# mul = 1 pour afficher une img .jpeg ou jpg dans sa couleur d'origine
# mul = 255 pour afficher une img .png dans sa couleur d'origine
multiplicateur = 1

if (os.path.isfile("./res/" + prefix + ".jpeg")):
    filename = prefix + ".jpeg"
elif(os.path.isfile("./res/" + prefix + ".jpg")):
    filename = prefix + ".jpg"
elif(os.path.isfile("./res/" + prefix + ".png")):
    filename = prefix + ".png"
    multiplicateur = 255
elif prefix == "-help":
    exit()
else:
    print("il n'y a pas de fichier " + prefix +
          " .jpg, .jpeg ou .png, utilisez le program avec '-help' pour un detail d'utilisation")
    exit()

print(filename)

# recupere l'img
img = (mplimg.imread("res/" + filename).copy()
       * multiplicateur).astype(np.uint8)

functions.otsu(img)

plt.figure()
plt.title(filename)
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
