import cv2
import os
import numpy as np

from consts import *
import multires

# Charge les images d'un répertoire
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
        else:
            print("Erreur: n'a pas trouvé l'image", filename)
    return images

# Fonction principale
def main():
    # Charge les images
    dataset = load_images(IN_DIR)
    if dataset:
        # Affiche la première image
        cv2.imshow("Premiere image", dataset[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convertit les images en niveaux de gris
    dataset = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    if dataset:
        # Affiche la première image
        cv2.imshow("Premiere image en niveau de gris", dataset[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Génère les images multirésolution
    image_pyramid = multires.run(dataset, LEVELS)
    if image_pyramid:
        # Affiche les dimensions de la première image à chaque niveau
        for level, images_at_level in enumerate(image_pyramid):
            print(f"Niveau {level}:")
            print(f"  Image 0: Dimensions {images_at_level[0].shape}")

main()
