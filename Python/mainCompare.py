import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

from consts import *
import utils
import pyramideV2
import illumination

def compare_images(imgA, imgB):
    imgA_pro, imgB_pro = illumination.illumination_correction(imgA, imgB)
    return ssim(imgA, imgB), ssim(imgA_pro, imgB_pro)

if __name__ == '__main__':
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
    dataset_used = pyramideV2.last_mean_pyramid(NB_PIXELS_MIN, dataset_cropped)

    # Mesure du temps d'exécution
    pyramid_time = time.time()
    print(f"Pyramide d'images créée en : {pyramid_time - crop_time:.2f} secondes")

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_used[0])

    # Ajuster toutes les images du dataset
    adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_used]

    # Mesure du temps d'exécution
    adjust_time = time.time()
    print(f"Luminosité images ajustées en : {adjust_time - pyramid_time:.2f} secondes", end="\n\n")

    # Comparer les images
    resultsv1 = np.zeros(nb_images - 1)
    resultsv2 = np.zeros(nb_images - 1)
    for i in range(1, nb_images):
        resultsv1[i - 1], resultsv2[i - 1] = compare_images(adjusted_dataset[i], adjusted_dataset[i-1])
    variationsv1 = np.diff(resultsv1)
    variationsv2 = np.diff(resultsv2)
    # Mesure du temps d'exécution
    compare_time = time.time()
    print(f"Comparaison des images en : {compare_time - adjust_time:.2f} secondes")

    # Afficher les résultats
    plt.figure()
    plt.plot(resultsv1)
    plt.xlabel("Image i comparée à l'image i-1")
    plt.ylabel("Similarité structurale")
    plt.title("Comparaison des images v1")

    plt.figure()
    plt.plot(resultsv2)
    plt.xlabel("Image i comparée à l'image i-1")
    plt.ylabel("Similarité structurale")
    plt.title("Comparaison des images v2")

    plt.figure()
    plt.plot(variationsv1)
    plt.xlabel("Image i comparée à l'image i-1")
    plt.ylabel("Variation de similarité structurale")
    plt.title("Variation de similarité structurale v1")

    plt.figure()
    plt.plot(variationsv2)
    plt.xlabel("Image i comparée à l'image i-1")
    plt.ylabel("Variation de similarité structurale")
    plt.title("Variation de similarité structurale v2")
    plt.show()
