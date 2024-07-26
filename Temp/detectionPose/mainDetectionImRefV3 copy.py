import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # Reconstruire la matrice I avec les valeurs singulières tronquées
    I_trunc = U @ np.diag(S_trunc) @ Vt

    # Déterminer l'écart entre les deux matrices
    diff = np.linalg.norm(I - I_trunc)
    return diff



def detect_pose_transition(diff_values, start_index_a_verif, threshold=0.5, post_threshold_factor=1.2, percentage_stable=0.66, num_images_after=3):
    """
    Détecte une transition de pose dans une séquence d'images
    en comparant les différences entre les images.
    """

    # Calculer les différences entre les valeurs adjacentes valides
    diff_adjacent = (np.diff(diff_values))

    # Normalisation min-max des différences adjacentes entre 0 et 1
    min_diff = np.min(diff_adjacent)
    max_diff = np.max(diff_adjacent)
    if min_diff == max_diff:
        normalized_diff_adjacent = np.zeros_like(diff_adjacent)
    else:
        normalized_diff_adjacent = (diff_adjacent - min_diff) / (max_diff - min_diff)

    plt.figure()
    plt.plot(np.arange(start_index_a_verif, start_index_a_verif + len(normalized_diff_adjacent)), normalized_diff_adjacent, 'o-', label='Différences adjacentes')
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
    # plt.show()

    # Rechercher le premier indice où la différence dépasse le seuil
    transition_indices = np.where(normalized_diff_adjacent > threshold)[0]

    # Vérifier si une transition potentiel a été détectée
    if len(transition_indices) == 0:
        return None
    
    best_transition_index = None
    best_stable_percentage = 0.0

    # Tester chaque indice de transition potentiel
    for idx in transition_indices:
    # Vérification de la stabilité sur un nombre fixe d'images après le point de transition
        end_index = idx + num_images_after + 1
        if end_index > len(diff_values):
            break

        post_transition_values = diff_values[idx + 1:end_index]
        post_mean = np.mean(post_transition_values) 

        # Vérifier si au moins percentage_stable des valeurs post-transition sont stables
        num_stable_values = np.sum(np.abs(post_transition_values - post_mean) < post_threshold_factor * np.std(post_transition_values))
        stability_percentage = num_stable_values / num_images_after

        # Mettre à jour le meilleur indice de transition si le pourcentage est plus élevé
        if stability_percentage > best_stable_percentage:
            best_transition_index = idx
            best_stable_percentage = stability_percentage

    # Vérifier si le meilleur pourcentage est supérieur au seuil minimum requis
    if best_stable_percentage >= percentage_stable:
        return best_transition_index + start_index_a_verif + 1 # +1 car np.diff réduit la taille de 1
    else:
        return None


def step(dataset, start_index):
    nb_images = len(dataset)
    if nb_images - start_index < 8: # Il ne peut pas y avoir de transition de pose
        return None, None
    print("Nombre d'images dans le dataset donné dans step :", nb_images - start_index)
    print("Start index dans step :", start_index)

    images_ref = dataset[start_index:start_index+3]
    diff_values = np.zeros(nb_images - (start_index + 3))  # Liste pour stocker les valeurs de diff
    
    for i in range(len(diff_values)):
        diff_values[i] = process_4_images(images_ref + [dataset[i + start_index + 3]])

    # Détection de la première transition de pose
    transition_index = detect_pose_transition(diff_values, start_index + 3)
    
    return diff_values, transition_index

def main():
    """
    On considère :
        qu'on a au moins 4 images pour une même pose.
        qu'on peux aller au niveau 4 de la pyramide.
    """
    # Charger les images
    dataset = utils.load_images(IN_DIR)
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    nb_images = len(dataset)

    # Vérifier qu'on a au moins 4 images (avec 3 de référence)
    if nb_images < 4:
        print("Erreur: le dataset doit contenir au moins 4 images.")
        return
    print(nb_images, "images chargées.")

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_gray[0])

    # Ajuster toutes les images du dataset
    adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_gray]

    # Créer la pyramide d'images
    dataset_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, adjusted_dataset)

    # On utilise le niveau 4 pour les calculs
    dataset_used = (dataset_pyramid[4]).tolist()

    # Boucle principale
    num_img_ref = 0
    ind_changes = []

    while num_img_ref is not None and num_img_ref < nb_images - 3:
        diff_values, transition_index = step(dataset_used, num_img_ref)
        
        if transition_index is not None and transition_index != num_img_ref:
            ind_changes.append(transition_index)
            num_img_ref = transition_index
        else:
            break
        # plt.figure()
        # plt.plot(diff_values, 'o-')
        # plt.title('Différences')
        # plt.xlabel('Index')
        # plt.ylabel('Différence')
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)


    print("Indices des changements de pose : ", ind_changes)
    # plt.show()

main()