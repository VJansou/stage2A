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



def detect_pose_transition(diff_values, start_index, threshold=0.6, post_threshold_factor=1.5, percentage_stable=0.66, num_images_after=3):
    """
    Détecte une transition de pose dans une séquence d'images
    en comparant les différences entre les images.
    """
    # Vérifier si toutes les valeurs sont NaN
    if np.all(np.isnan(diff_values)):
        return None

    # Créer un masque pour les valeurs non NaN
    valid_mask = np.isfinite(diff_values)
    valid_diff_values = diff_values[valid_mask]

    # Calculer les différences entre les valeurs adjacentes valides
    diff_adjacent = np.abs(np.diff(valid_diff_values))

    # Vérifier si les différences adjacentes sont non vides
    if len(diff_adjacent) == 0:
        return None

    # Normalisation min-max des différences adjacentes entre 0 et 1
    min_diff = np.min(diff_adjacent)
    max_diff = np.max(diff_adjacent)
    if min_diff == max_diff:
        normalized_diff_adjacent = np.zeros_like(diff_adjacent)
    else:
        normalized_diff_adjacent = (diff_adjacent - min_diff) / (max_diff - min_diff)

    plt.figure()
    plt.plot(np.arange(start_index + 3, start_index + 3 + len(normalized_diff_adjacent)), normalized_diff_adjacent, 'o-', label='Différences adjacentes')
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Changement de pose réel') # Changement de pose réel
    # plt.show()

    # Rechercher le premier indice où la différence dépasse le seuil
    transition_indices = np.where(normalized_diff_adjacent > threshold)[0]

    # Vérifier si une transition a été détectée
    if len(transition_indices) == 0:
        return None
    
    best_transition_index = None
    best_stable_percentage = 0.0

   # Tester chaque indice de transition potentiel
    for idx in transition_indices:
        potential_transition = idx + 1  # +1 car np.diff réduit la taille de 1
        mapped_transition_index = np.where(valid_mask)[0][potential_transition]

    # Vérification de la stabilité sur un nombre fixe d'images après le point de transition
        end_index = mapped_transition_index + num_images_after + 1  # +1 pour inclure la dernière image à vérifier
        post_transition_values = diff_values[mapped_transition_index + 1:end_index]
        post_transition_valid_mask = np.isfinite(post_transition_values)
        valid_post_transition_values = post_transition_values[post_transition_valid_mask]

        if len(valid_post_transition_values) == 0:
            continue

        post_mean = np.mean(valid_post_transition_values)

        # Vérifier si au moins percentage_stable des valeurs post-transition sont stables
        num_stable_values = np.sum(np.abs(valid_post_transition_values - post_mean) < post_threshold_factor * np.std(valid_post_transition_values))
        stability_percentage = num_stable_values / len(valid_post_transition_values)

        # Mettre à jour le meilleur indice de transition si le pourcentage est plus élevé
        if stability_percentage > best_stable_percentage:
            best_transition_index = mapped_transition_index
            best_stable_percentage = stability_percentage

    # Vérifier si le meilleur pourcentage est supérieur au seuil minimum requis
    if best_stable_percentage >= percentage_stable:
        return best_transition_index + 1 # +1 pour obtenir l'indice de la première image après la transition
    else:
        return None


def step(dataset, start_index):
    nb_images = len(dataset)
    if nb_images - start_index < 6:
        return None, None
    print("Nombre d'images dans le dataset donné dans step :", nb_images - start_index)
    print("Start index dans step :", start_index)

    images_ref = dataset[start_index:start_index+3]
    diff_values = np.zeros(nb_images - start_index)  # Liste pour stocker les valeurs de diff
    diff_values[:3] = np.nan # Ignorer les 3 premières valeurs
    
    for num_image in range(start_index + 3, nb_images):
        diff_values[num_image - start_index] = process_4_images(images_ref + [dataset[num_image]])
    
    # Détection de la première transition de pose
    transition_index = detect_pose_transition(diff_values, start_index)
    
    if transition_index is not None:
        transition_index += start_index
    
    return diff_values, transition_index

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
    plt.show()

main()