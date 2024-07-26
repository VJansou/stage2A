import cv2
import numpy as np

from consts import *
import utils
import pyramide

def process_4_images(dataset):
    # Vérifier que le dataset contient 4 images
    if len(dataset) != 4:
        print("Erreur: le dataset doit contenir 4 images.")
        return None
    
    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]

    # Créer la pyramide d'images
    image_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, dataset_gray)

    # Créer la matrice I pour le niveau 3
    I = utils.create_I_matrix(image_pyramid[3])

    # Calculer la SVD de la matrice I
    U, S, Vt = np.linalg.svd(I, full_matrices=False)

    # Tronquer les valeurs singulières
    S_trunc = np.zeros_like(S)
    S_trunc[:1] = S[:1]

    # Reconstruire la matrice I avec les valeurs singulières tronquées
    I_trunc = U @ np.diag(S_trunc) @ Vt

    # Déterminer si l'écart entre les deux matrices dépasse le seuil
    diff = np.linalg.norm(I - I_trunc) / np.linalg.norm(I)
    return diff > SEUIL

def main():
    """
    On considère :
        qu'on a au moins 3 images pour une même pose.
        qu'on peux aller au niveau 3 de la pyramide.
    """
    # Charger les images
    dataset = utils.load_images(IN_DIR)
    nb_images = len(dataset)
    print(nb_images, "images chargées.")

    # Vérifier qu'on a au moins 3 images
    if nb_images < 3:
        print("Erreur: le dataset doit contenir au moins 3 images.")
        return
    
    # Boucle principale
    continue_processing = True
    num_changement_pose = []
    num_image = 0
    while continue_processing:
        # Sélectionner 4 images si possible
        if num_image + 4 <= nb_images:
            images = dataset[num_image:num_image + 4]

            # Vérifier si les 4 images correspondent à la même pose
            if process_4_images(images):
                num_changement_pose.append(num_image + 3)
                num_image += 4
            else:
                num_image += 1
        else:
            continue_processing = False
    
    # Afficher les indices des images où il y a un changement de pose
    print("Indices des images où il y a un changement de pose:", num_changement_pose)

main()
                


