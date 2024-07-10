import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from consts import *
import utils
import pyramide

def process_4_images(dataset):
    # Vérifier que le dataset contient 4 images
    if len(dataset) != 4:
        print("Erreur: le dataset doit contenir 4 images. Il en contient", len(dataset), ".")
        return None

    # Créer la matrice I pour le niveau 4
    I = utils.create_I_matrix(np.array(dataset))

    # Calculer la SVD de la matrice I
    U, S, Vt = np.linalg.svd(I, full_matrices=False)

    # Tronquer les valeurs singulières
    S_trunc = np.zeros_like(S)
    S_trunc[:3] = S[:3]
    print("S", S)
    print("S_trunc", S_trunc)

    # Reconstruire la matrice I avec les valeurs singulières tronquées
    I_trunc = U @ np.diag(S_trunc) @ Vt

    # Déterminer l'écart entre les deux matrices
    diff = np.linalg.norm(I - I_trunc)
    return diff

def main():
    """
    On considère :
        qu'on a au moins 4 images pour une même pose.
        qu'on peux aller au niveau 4 de la pyramide.

        Dans cette version, le but est de faire une rétro-validation avant de passer à l'étape suivante.
        Pour ça, on va calculer les différences entre les images qu'on pense être de la même pose en prenant toutes les combinaisons possibles.
        On va alors vérifier si la différence entre les images est faible. (déterminer faible ??)
        On peut vérifier avec les prochaines images si la différence est toujours élevée.
    """
    # Charger les images
    dataset = utils.load_images(IN_DIR)
    nb_images = len(dataset)

    # Vérifier qu'on a au moins 4 images (avec 3 de référence)
    if nb_images < 4:
        print("Erreur: le dataset doit contenir au moins 4 images.")
        return
    print(nb_images, "images chargées.")

    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]

    # Recadrer les images
    dataset_cropped = [utils.crop_image(img, CROP_FACTOR) for img in dataset_gray]

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_gray[0])

     # Ajuster toutes les images du dataset
    adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_cropped]

    # Créer la pyramide d'images
    dataset_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, adjusted_dataset)

    # On utilise un seul niveau pour les calculs
    dataset_used = (dataset_pyramid[LEVEL_USED]).tolist()

    diff_value = process_4_images(dataset_used[:3] + [dataset_used[5]])

    print("Différence entre les images :", diff_value)

main()


def step(dataset, start_index):

    nb_images = len(dataset)
    if nb_images - start_index < 8: # Il ne peut pas y avoir de transition de pose
        return None
    print("Nombre d'images dans le dataset donné dans step :", nb_images - start_index)
    print("Start index dans step :", start_index)

    images_ref = dataset[start_index:start_index+3]
    diff_values = np.zeros(nb_images - (start_index + 3))  # Liste pour stocker les valeurs de diff
    
    for i in range(len(diff_values)):
        diff_values[i] = process_4_images(images_ref + [dataset[i + start_index + 3]])

    # Détection de la première transition de pose
    transition_index_temp = detect_pose_transition(diff_values, start_index + 3)
    print("Transition index temp :", transition_index_temp)

    # Rétro-validation
    transition_index = transition_index_temp
    if transition_index_temp is not None:
        # Combinaisons pour les 3 images de référence parmi celles dont on est persuadé qu'elles sont de la même pose
        combinations_indices = list(combinations(range(start_index, transition_index_temp), 3))

        # Initialisation de la matrice de différences
        diff_valid = np.zeros((len(combinations_indices), nb_images - start_index))
        diff_valid[0,:3] = np.nan
        diff_valid[0,3:] = diff_values # On ne recalcule pas les différences pour les images déjà calculées

        # Calcul des différences pour les combinaisons
        for i, comb in enumerate(combinations_indices[1:]):
            images_ref = [dataset[j] for j in comb]
            for k in range(transition_index_temp, nb_images):
                if k in comb:
                    diff_valid[i,k - transition_index_temp] = np.nan
                else:
                    diff_valid[i,k - transition_index_temp] = process_4_images(images_ref + [dataset[k]])
        
        # plt.figure()
        # plt.imshow(diff_valid, cmap='hot', interpolation='nearest')
        # plt.colorbar()

        # Calcul de la moyenne et écart-type des différences dont on est persuadé qu'elle sont de la même pose
        mean_diff = np.nanmean(diff_valid[:, :transition_index_temp - start_index])
        std_diff = np.nanstd(diff_valid[:, :transition_index_temp - start_index])
        print("Mean diff :", mean_diff)
        print("Std diff :", std_diff)

        # Calcul du seuil
        threshold = mean_diff + std_diff # Contient ~ 68% des valeurs
    
        # Moyenne des différences pour chaque image suivante
        mean_diff_valid = np.mean(diff_valid[:, transition_index_temp - start_index:], axis=0)
        print("Mean diff valid :", mean_diff_valid, mean_diff_valid.dtype)
        # plt.figure()
        # plt.plot(mean_diff_valid, 'o-', label='Moyenne des différences')
        # plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
        # plt.show()

        # Maj de la transition de pose

        nb_tentative = 3 # On se donne 3 tentatives pour trouver une transition de pose
        idx = 0
        while nb_tentative > 0 and idx < len(mean_diff_valid):
            print(mean_diff_valid[idx])
            if mean_diff_valid[idx] < threshold: # On considère que l'image est dans la même pose que les autres
                transition_index += (4 - nb_tentative)
                nb_tentative = 3 # On réinitialise le nombre de tentatives
            else:
                nb_tentative -= 1
            idx += 1
    print("Transition index :", transition_index)
    return transition_index