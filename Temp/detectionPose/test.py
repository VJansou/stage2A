import pyramide
import pyramideV2
import utils
from consts import *
import cv2
import matplotlib.pyplot as plt

# Test pyramideV2
if __name__ == "__main__":

    # Charger les images
    dataset = utils.load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)

    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]

    # Recadrer les images
    dataset_cropped = [utils.crop_image(img, CROP_FACTOR) for img in dataset_gray]

    # Créer la pyramide d'images
    dataset_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, dataset_cropped)

    # Avec la v2
    last_images_pyramid = pyramideV2.last_mean_pyramid(NB_PIXELS_MIN, dataset_cropped)

    # Afficher les premières images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(dataset_pyramid[LEVEL_USED][0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(last_images_pyramid[0], cmap='gray')
    plt.show()
    