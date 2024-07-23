import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

from itertools import combinations
from consts import *
import utils
import pyramide
import pyramideV2

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

def process_4_images_pixel(dataset, seuil):
    """
    Fait une comparaison pixel par pixel.
    Renvoie le nombre de pixels considérés comme différents.
    """
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

    # Déterminer l'écart pour chaque pixel
    diff = np.linalg.norm(I - I_trunc, axis=1)




def detect_pose_transition(diff_values, start_index_a_verif, threshold=0.3, post_threshold=0.4, percentage_stable=0.66, num_images_after=3):
    """
    Détecte une transition de pose dans une séquence d'images
    en comparant les différences entre les images.
    """
    # Calculer les différences entre les valeurs adjacentes valides
    diff_adjacent = np.abs((np.diff(diff_values)))

    # Normalisation min-max des différences adjacentes entre 0 et 1
    min_diff = np.min(diff_adjacent)
    max_diff = np.max(diff_adjacent)
    if min_diff == max_diff:
        normalized_diff_adjacent = np.zeros_like(diff_adjacent)
    else:
        normalized_diff_adjacent = (diff_adjacent - min_diff) / (max_diff - min_diff)

    # plt.figure()
    # plt.plot(np.arange(start_index_a_verif, start_index_a_verif + len(normalized_diff_adjacent)), normalized_diff_adjacent, 'o-', label='Différences adjacentes')
    # plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
    # plt.show()

    # Renvoyer le premier indice où la valeur est supérieure au seuil
    # return np.argmax(normalized_diff_adjacent > np.mean(normalized_diff_adjacent) + np.std(normalized_diff_adjacent)) + start_index_a_verif + 1

    # Rechercher le premier indice où la différence dépasse le seuil
    transition_indices = np.where(normalized_diff_adjacent > threshold)[0]

    # Vérifier si une transition potentiel a été détectée
    if len(transition_indices) == 0:
        return None
    # print("Transition indices :", transition_indices)
    return transition_indices[0] + start_index_a_verif + 1
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
        print("Post transition values :", post_transition_values)
        print("Post mean :", post_mean)

        # Vérifier si au moins percentage_stable des valeurs post-transition sont stables
        num_stable_values = np.sum(np.abs(post_transition_values - post_mean) < post_threshold)
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
        return None
    # print("Nombre d'images dans step :", nb_images - start_index)
    # print("Start index dans step :", start_index)

    images_ref = dataset[start_index:start_index+3]
    diff_values = np.zeros(nb_images - (start_index + 3))  # Vecteur pour stocker les valeurs de diff

    for i in range(len(diff_values)):
        diff_values[i] = process_4_images(images_ref + [dataset[i + start_index + 3]])
    
    # Détection de la première transition de pose
    transition_index_temp = detect_pose_transition(diff_values, start_index + 3)
    print("Transition index temp :", transition_index_temp)

    # Rétro-validation
    transition_index = transition_index_temp
    if transition_index_temp is not None:
        # Combinaisons pour les 3 images de référence parmi celles dont on est persuadé qu'elles sont de la même pose
        combinaisons_indices = list(combinations(range(start_index, transition_index_temp), 3))

        # Initialisation de la matrice de différences
        diff_valid = np.zeros((len(combinaisons_indices), nb_images - start_index))
        diff_valid[0,:3] = np.nan
        diff_valid[0,3:] = diff_values
        # print(start_index, combinaisons_indices[0])

        # Calcul des différences pour les combinaisons
        for i, comb in enumerate(combinaisons_indices[1:]):
            images_ref = [dataset[j] for j in comb]
            for k in range(nb_images - start_index):
                if k + start_index in comb:
                    diff_valid[i+1,k] = np.nan
                else:
                    diff_valid[i+1,k] = process_4_images(images_ref + [dataset[k + start_index]])

        # Calcul de la moyenne et écart-type des différences dont on est persuadé qu'elle sont de la même pose
        mean_diff = np.nanmean(diff_valid[:, :transition_index_temp - start_index])
        std_diff = np.nanstd(diff_valid[:, :transition_index_temp - start_index])

        # Autre méthode avec la médiane et l'IQR
        # med_diff = np.nanmedian(diff_valid[:, :transition_index_temp - start_index])
        # print("test", transition_index_temp - start_index)
        # print(diff_valid[:, :transition_index_temp - start_index])
        # print("Median diff :", med_diff)
        # med_diff_sorted = np.sort(med_diff)
        # Q1 = np.percentile(med_diff_sorted, 25)
        # Q3 = np.percentile(med_diff_sorted, 75)
        # IQR = Q3 - Q1
        # threshold = Q3 + 1.5 * IQR

        # Calcul du seuil
        threshold = mean_diff + std_diff # Contient ~ 68% des valeurs
        # print("Threshold :", threshold)

        # Médiane des différences pour chaque image suivante
        med_diff_valid = np.median(diff_valid[:, transition_index_temp - start_index:], axis=0)
        # print("Mean diff valid :", med_diff_valid, med_diff_valid.dtype)

        # Calcul de l'IQR
        med_diff_valid_sorted = np.sort(med_diff_valid)
        Q1 = np.percentile(med_diff_valid_sorted, 25)
        Q3 = np.percentile(med_diff_valid_sorted, 75)

        IQR = Q3 - Q1

        # Seuil pour la détection de transition
        lower_bound = Q1 - 3 * IQR
        # print("Lower bound :", lower_bound)

        # Maj de la transition de pose
        nb_tentative = 2 # On se donne 3 tentatives pour trouver une transition de pose
        idx = 0
        while nb_tentative > 0 and idx < len(med_diff_valid):
            if med_diff_valid[idx] < lower_bound:
            # if med_diff_valid[idx] <= threshold or med_diff_valid[idx] < lower_bound: # On considère que l'image est dans la même pose que les autres
                transition_index += 3 - nb_tentative
                nb_tentative = 2 # On réintialise le nombre de tentatives
            else:
                nb_tentative -= 1
            idx += 1
        # print("Transition index :", transition_index)
        return transition_index
                
           

if __name__ == "__main__":
    """
    On considère :
        qu'on a au moins 4 images pour une même pose.
        qu'on peux aller au niveau 4 de la pyramide.

        Dans cette version, le but est de faire une rétro-validation avant de passer à l'étape suivante.
        Pour ça, on va calculer les différences entre les images qu'on pense être de la même pose en prenant toutes les combinaisons possibles.
        On va alors vérifier si la différence entre les images est faible. (déterminer faible ??)
        On peut vérifier avec les prochaines images si la différence est toujours élevée.
    """
    # Mesure du temps d'exécution
    start_time = time.time()

    # Charger les images
    dataset = utils.load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)

    # Vérifier qu'on a au moins 4 images (avec 3 de référence)
    if nb_images < 4:
        raise ValueError("Erreur: le dataset doit contenir au moins 4 images.")
    
    # Mesure du temps d'exécution
    load_time = time.time()
    print(f"{nb_images} images chargées en : {load_time - start_time:.2f} secondes")

    # Trier les images par date de prise de vue
    dataset_sorted = utils.sort_images_by_date(dataset)

    # Mesure du temps d'exécution
    sort_time = time.time()
    print(f"Images triées par date de prise de vue en : {sort_time - load_time:.2f} secondes")

    # Convertir les images en niveaux de gris
    # dataset_gray = utils.convert_to_gray(dataset)
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset_sorted]

    # Mesure du temps d'exécution
    gray_time = time.time()
    print(f"Images converties en niveaux de gris en : {gray_time - sort_time:.2f} secondes")

    # Recadrer les images
    dataset_cropped = [utils.crop_image(img, CROP_FACTOR) for img in dataset_gray]

    # Mesure du temps d'exécution
    crop_time = time.time()
    print(f"Images recadrées en : {crop_time - gray_time:.2f} secondes")

    # # Créer la pyramide d'images
    # dataset_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, dataset_cropped)
    # # pyramide.show_first_image(dataset_pyramid)

    # # On utilise un seul niveau pour les calculs
    # dataset_used = (dataset_pyramid[LEVEL_USED]).tolist()
    dataset_used = pyramideV2.last_mean_pyramid(NB_PIXELS_MIN, dataset_cropped)

    # Mesure du temps d'exécution
    pyramid_time = time.time()
    print(f"Pyramide d'images créée en : {pyramid_time - crop_time:.2f} secondes")

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_used[0])

    # Ajuster toutes les images du dataset
    adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_used]
    # adjusted_dataset = utils.adjust_images_mean(dataset_used)
    # adjusted_dataset = dataset_used
    # Mesure du temps d'exécution
    adjust_time = time.time()
    print(f"Luminosité images ajustées en : {adjust_time - pyramid_time:.2f} secondes", end="\n\n")

    # Boucle principale
    num_img_ref = 0
    ind_changes = []

    while num_img_ref is not None and num_img_ref < nb_images - 3:
        transition_index = step(adjusted_dataset, num_img_ref)
        
        if transition_index is not None and transition_index < nb_images - 3:
            ind_changes.append(transition_index)
            num_img_ref = transition_index
        else:
            break

    print("Indices des changements de pose : ", ind_changes)

    # Mesure du temps d'exécution
    boucle_time = time.time()
    print(f"Boucle principale terminée en : {boucle_time - pyramid_time:.2f} secondes", end="\n\n")

    # plt.show()

    # Mesure du temps d'exécution
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution: {elapsed_time:.2f} secondes", end="\n\n")