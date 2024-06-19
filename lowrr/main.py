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

debugging = False
# Fonction principale
def main():
    # Charge les images
    dataset = load_images(IN_DIR)
    nb_images = len(dataset)
    if debugging:
        # Affiche la première image
        cv2.imshow("Premiere image", dataset[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convertit les images en niveaux de gris
    dataset = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    if debugging:
        # Affiche la première image
        cv2.imshow("Premiere image en niveau de gris", dataset[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Génère les images multirésolution
    image_pyramid = multires.run(dataset, LEVELS)
    if debugging:
        # Affiche les dimensions de la première image à chaque niveau
        for level, images_at_level in enumerate(image_pyramid):
            print(f"Niveau {level}:")
            print(f"  Image 0: Dimensions {images_at_level[0].shape}")
        multires.show_first_image(image_pyramid)
    
    # Initialisation du vecteur de mouvement
    motion_vector = np.zeros([nb_images,6])

    # Algorithme multi-résolution.
    # Effectue la même opération à chaque niveau pour les images et gradients correspondants.
    # L'itérateur est inversé pour commencer par le dernier niveau (la résolution la plus basse).
    # Le niveau 0 correspond aux images initiales.
    image_pyramid_inv = image_pyramid[::-1]

    for level, images_at_level in enumerate(image_pyramid_inv):
        (rows, cols) = images_at_level[0].shape

        # Adaptation du vecteur de mouvement au changement de résolution
        motion_vector[:,4] *= 2
        motion_vector[:,5] *= 2

        # Variables d'état pour la boucle
        nb_iter = 0
        old_imgs_a = np.zeros([nb_images, rows*cols])
        errors = np.zeros([nb_images, rows*cols])
        lagrange_mult_rho = np.zeros([nb_images, rows*cols])

        # Boucle principale
        continue_loop = True
        while continue_loop:
            # Pre-scale lambda
            lambda_value = LAMBDA/np.sqrt(rows)

            # mise à jour de A avec l'approximation de faible rang

            # mise à jour de e : L1-regularized least-squares

            # mise à jour de theta : forwards compositional step of a Gauss-Newton approximation.
main()
