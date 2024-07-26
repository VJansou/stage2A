import matplotlib.pyplot as plt
import cv2
import numpy as np

import utils
import pyramideV2
from consts import *
import illumination

if __name__ == "__main__":
    # Charger les images
    dataset = utils.load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)
    dataset = dataset[:2]

    # Trie les images par date de prise de vue
    dataset_sorted = utils.sort_images_by_date(dataset)

    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset_sorted]

    # Recadrer les images
    dataset_cropped = [utils.crop_image(img, CROP_FACTOR) for img in dataset_gray]

    # Créer la pyramide d'images
    dataset_used = pyramideV2.last_mean_pyramid(NB_PIXELS_MIN, dataset_cropped)

    # Ajuster les deux premières images
    img1, img2 = dataset_used[:2]
    img1_adjusted, img2_adjusted = illumination.illumination_correction(img1, img2)

    # Afficher les images
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Image 1")
    plt.subplot(2, 2, 2)
    plt.imshow(img1_adjusted, cmap='gray')
    plt.title("Image 1 ajustée")
    plt.subplot(2, 2, 3)
    plt.imshow(img2, cmap='gray')
    plt.title("Image 2")
    plt.subplot(2, 2, 4)
    plt.imshow(img2_adjusted, cmap='gray')
    plt.title("Image 2 ajustée")
    plt.show()


    # Comparer les images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(img1 - img1_adjusted), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(img2 - img2_adjusted), cmap='gray')
    plt.show()
