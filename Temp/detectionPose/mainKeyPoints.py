import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

import utils
from consts import *
import vueGraphique

def extract_keypoints(image):
    """
    Retourne les keypoints et leurs coordonnées pour une image donnée.
    """
    orb = cv2.ORB.create()
    keypoints = orb.detect(image, None)
    descriptors = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2, max_distance=1.0):
    """
    Retourne les correspondances entre les keypoints de deux images.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return []
    matches = bf.match(descriptors1, descriptors2)
    matches = [match for match in matches if match.distance < max_distance]
    return matches

def calculate_matches(images, max_distance=1.0):
    """
    Calcule le nombre de correspondances entre les keypoints de chaque paire d'images consécutives.
    """
    matches_count = np.zeros(len(images) - 1)
    for i in range(1, len(images)):
        keypoints1, descriptors1 = extract_keypoints(images[i-1])
        keypoints2, descriptors2 = extract_keypoints(images[i])
        
        matches = match_keypoints(descriptors1, descriptors2, max_distance)
        # Affichage des correspondances
        # display_matches(images[i-1], keypoints1, images[i], keypoints2, matches)
        
        print(f"Nombre de correspondances entre les images {i-1} et {i}:", len(matches))
        matches_count[i-1] = len(matches)
    return matches_count

def detect_view_changes_factor(matches_count, factor=0.5):
    """
    Détecte les changements de vue en fonction du nombre de correspondances entre les images.
    Version avec un facteur de réduction du nombre de correspondances.
    """
    changes = []
    for i in range(len(matches_count)):
        if matches_count[i] < matches_count[i-1] * factor:
            changes.append(i + 1)
    return changes

def detect_view_changes_mean(matches_count, mean_factor=1.0):
    """
    Détecte les changements de vue en fonction du nombre de correspondances entre les images.
    Version où on compare le nombre de correspondances à la moyenne.
    """
    changes = []
    mean = np.mean(matches_count)
    std = np.std(matches_count)
    seuil = mean - std * mean_factor
    print(f"Moyenne: {mean:.2f}")
    print(f"Écart-type: {std:.2f}")
    print(f"Seuil de changement de vue: {seuil:.2f}")
    for i in range(len(matches_count)):
        if matches_count[i] < seuil:
            changes.append(i + 1)
    return changes

def detect_view_changes_mediane(matches_count, median_factor=1.0):
    """
    Détecte les changements de vue en fonction du nombre de correspondances entre les images.
    Version où on compare le nombre de correspondances à la médiane et l'IQR.
    """
    changes = []
    mediane = np.median(matches_count)
    q1 = np.percentile(matches_count, 25)
    q3 = np.percentile(matches_count, 75)
    iqr = q3 - q1
    seuil = mediane - iqr * median_factor
    if seuil < 0:
        seuil = q1
    print(f"Médiane: {mediane:.2f}")
    print(f"q1: {q1:.2f}")
    print(f"q3: {q3:.2f}")
    print(f"Seuil de changement de vue: {seuil:.2f}")
    for i in range(len(matches_count)):
        if matches_count[i] < seuil:
            changes.append(i + 1)
    return changes

def display_matches(img1, keypoints1, img2, keypoints2, matches):
    """
    Affiche les correspondances entre deux images.
    """
    matched_image = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Correspondances", matched_image)
    cv2.waitKey(0)  # Attente d'une touche pour continuer
    cv2.destroyAllWindows()

def user_verification(images, changes):
    """
    Vérification manuelle des changements de vue détectés.
    """
    confirmed_changes = []
    for idx in changes:
        img1 = images[idx-1]
        img2 = images[idx]
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img1)
        plt.subplot(1,2,2)
        plt.imshow(img2)
        plt.show()

        # cv2.imshow("Image 1", img1)
        # cv2.imshow("Image 2", img2)
        print(f"Changement de vue détecté entre les images {idx-1} et {idx}.")
        response = input("Confirmer le changement de vue (o/n) ?")
        if response.lower() == "o":
            print("Changement de vue confirmé.")
            confirmed_changes.append(idx)
        else:
            print("Changement de vue non confirmé.")
        cv2.destroyAllWindows()
    return confirmed_changes

if __name__ == "__main__":
    # Mesure du temps d'exécution
    start_time = time.time()

    # Charger les images
    dataset = utils.load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)
    
    # Mesure du temps d'exécution
    load_time = time.time()
    print(f"{nb_images} images chargées en : {load_time - start_time:.2f} secondes")

    # Trier les images par date de prise de vue
    dataset_sorted = utils.sort_images_by_date(dataset)
    print("taille images", dataset_sorted[0].shape)

    # Mesure du temps d'exécution
    sort_time = time.time()
    print(f"Images triées par date de prise de vue en : {sort_time - load_time:.2f} secondes")

    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset_sorted]

    # Mesure du temps d'exécution
    gray_time = time.time()
    print(f"Images converties en niveaux de gris en : {gray_time - sort_time:.2f} secondes")

    # Calcul des correspondances
    matches_count = calculate_matches(dataset_gray)

    # Détection des changements de vue
    # changes = detect_view_changes_factor(matches_count, factor=0.5)
    changes = detect_view_changes_mediane(matches_count, median_factor=1.0)
    # changes = detect_view_changes_mean(matches_count, mean_factor=1.0)
    print()
    print("Changements de vue détectés aux images:", changes)

    # Mesure du temps d'exécution
    end_time = time.time()
    print(f"Changements de vue trouvés en : {end_time - gray_time:.2f} secondes")
    print()
    print(f"Temps total d'exécution : {end_time - start_time:.2f} secondes")
    # Vérification manuelle des changements de vue
    # if len(changes) > 0:
    #     confirmed_changes = vueGraphique.user_verification_interface(dataset_sorted, changes)
    #     print("Changements de vue confirmés aux images:", confirmed_changes)
    # else:
    #     print("Aucun changement de vue détecté.")