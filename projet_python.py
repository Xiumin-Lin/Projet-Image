import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import os.path

if len(sys.argv)<2:
    print("utilisez le program avec l'option '-help' pour un detail d'utilisation")
    exit()

prefix = sys.argv[1]

if (os.path.isfile("./pieces/" + prefix + ".jpeg")):
    filename = prefix + ".jpeg"
elif(os.path.isfile("./pieces/" + prefix + ".jpg")):
    filename = prefix + ".jpg"
elif(os.path.isfile("./pieces/" + prefix + ".png")):
    filename = prefix + ".png"
elif prefix == "-help":
    exit()
else: 
    print("il n'y a pas de fichier " + prefix + " .jpg, .jpeg ou .png, utilisez le program avec '-help' pour un detail d'utilisation")
    exit()

print(filename)