import cv2
import numpy as np
import time

import utils
from consts import *

def extract_features(image, method='ORB'):
    if method == 'ORB':
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
    elif method == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
    elif method == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, method='ORB'):
    if method == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method in ['SIFT', 'SURF']:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return matches

def calculate_matches(images, method='ORB'):
    matches_count = np.zeros(len(images) - 1)
    for i in range(1, len(images)):
        _, descriptors1 = extract_features(images[i-1], method)
        _, descriptors2 = extract_features(images[i], method)
        if descriptors1 is not None and descriptors2 is not None:
            matches = match_features(descriptors1, descriptors2, method)
            print(f"Nombre de correspondances entre les images {i-1} et {i}:", len(matches))
            matches_count[i-1] = len(matches)
    return matches_count

def detect_view_changes(matches_count, factor=0.5):
    changes = []
    for i in range(1, len(matches_count)):
        if matches_count[i] < matches_count[i-1] * factor:
            changes.append(i + 1)
    return changes


if __name__ == "__main__":
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
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset_sorted]

    # Mesure du temps d'exécution
    gray_time = time.time()
    print(f"Images converties en niveaux de gris en : {gray_time - sort_time:.2f} secondes")

    # Calcul des correspondances
    matches_count = calculate_matches(dataset_gray)

    # Détection des changements de vue
    changes = detect_view_changes(matches_count, factor=0.5)
    print("Changements de vue détectés aux images:", changes)

    # Mesure du temps d'exécution
    end_time = time.time()
    print(f"Changements de vue détectés en : {end_time - gray_time:.2f} secondes")
