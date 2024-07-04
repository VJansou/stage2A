import numpy as np
import matplotlib.pyplot as plt
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

# Fonction de test
def test_detect_pose_transition():
    # Données de test (différences simulées entre images)
    diff_values = np.array([0.1, 0.2, 0.8, 0.2, 0.3, 0.2, 0.1, 0.4, 0.2, 0.2, 0.1, 0.2])
    start_index = 0  # Index de départ (simulé)
    threshold = 0.5
    post_threshold_factor = 1.5
    percentage_stable = 0.6
    num_images_after = 3

    # Appel de la fonction à tester
    transition_index = detect_pose_transition(diff_values, start_index, threshold, post_threshold_factor, percentage_stable, num_images_after)

    # Affichage du résultat du test
    if transition_index is not None:
        print(f"Transition de pose détectée à l'index {transition_index}.")
    else:
        print("Aucune transition de pose détectée.")

    # Affichage graphique pour vérification visuelle
    plt.figure()
    plt.plot(np.arange(start_index + 3, start_index + 3 + len(diff_values)), diff_values, 'o-', label='Différences')
    if transition_index is not None:
        plt.axvline(x=transition_index, color='red', linestyle='--', linewidth=2, label='Transition de pose détectée')
    plt.xlabel('Index de l\'image de base')
    plt.ylabel('Différence')
    plt.legend()
    plt.grid(True)
    plt.show()

# Appel du test
test_detect_pose_transition()