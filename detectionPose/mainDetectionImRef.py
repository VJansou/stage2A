import cv2
import numpy as np
import matplotlib.pyplot as plt

from consts import *
import utils
import pyramide

def process_4_images(dataset):
    # Vérifier que le dataset contient 4 images
    if len(dataset) != 4:
        print("Erreur: le dataset doit contenir 4 images.")
        return None

    # Créer la pyramide d'images
    image_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, dataset)

    # Créer la matrice I pour le niveau 4
    I = utils.create_I_matrix(image_pyramid[4])

    # Calculer la SVD de la matrice I
    U, S, Vt = np.linalg.svd(I, full_matrices=False)

    # Tronquer les valeurs singulières
    S_trunc = np.zeros_like(S)
    S_trunc[:3] = S[:3]

    # Reconstruire la matrice I avec les valeurs singulières tronquées
    I_trunc = U @ np.diag(S_trunc) @ Vt

    # Déterminer l'écart entre les deux matrices
    diff = np.linalg.norm(I - I_trunc)
    return diff

def main():
    """
    On considère :
        qu'on a au moins 3 images pour une même pose.
        qu'on peux aller au niveau 4 de la pyramide.
    """
    # Charger les images
    dataset = utils.load_images(IN_DIR)
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    nb_images = len(dataset)
    print(nb_images, "images chargées.")

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_gray[0])

    # Ajuster toutes les images du dataset
    adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_gray]
    nb_images = len(adjusted_dataset)

    # Vérifier qu'on a au moins 4 images (avec 3 de référence)
    if nb_images < 4:
        print("Erreur: le dataset doit contenir au moins 4 images.")
        return
    images_ref = adjusted_dataset[:3]
    
    # Boucle principale
    continue_processing = True
    diff_values = []  # Liste pour stocker les valeurs de diff
    num_image = 3
    while continue_processing:
        # Sélectionner 1 image si possible
        if num_image < nb_images :

            # Calucler la différence entre les images
            diff_values.append(process_4_images(images_ref + [adjusted_dataset[num_image]]))
            num_image += 1
        else:
            continue_processing = False
    
    # Afficher les différences
    print("Valeurs de diff: ", diff_values)
    plt.figure()
    plt.plot(range(3, nb_images), diff_values, 'o-', label='Différences')
    plt.axvline(x=8, color='red', linestyle='--', linewidth=2, label='Changement de pose réel') # Changement de pose réel
    plt.title('Différences')
    plt.xlabel('Index')
    plt.ylabel('Différence')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
main()
                


