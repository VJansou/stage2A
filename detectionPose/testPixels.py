import numpy as np
import matplotlib.pyplot as plt
import cv2

from consts import *
import utils
import pyramideV2

def process_4_images_pixel(dataset, seuil):
    """
    Fait une comparaison pixel par pixel.
    Renvoie le nombre de pixels considérés comme différents.
    """
    # Vérifier que le dataset contient 4 images
    if len(dataset) != 4:
        print("Erreur: le dataset doit contenir 4 images. Il en contient", len(dataset), ".")
        return None
    
    row, col = dataset[0].shape

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
    diff = diff.reshape((row, col))
    return diff, diff > seuil

if __name__ == "__main__":
    # Charger les images
    dataset = utils.load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)

    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]

    # Recadrer les images
    dataset_cropped = [utils.crop_image(img, CROP_FACTOR) for img in dataset_gray]

    # Créer la pyramide d'images
    dataset_used = pyramideV2.last_mean_pyramid(NB_PIXELS_MIN, dataset_cropped)

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_used[0])

    # Ajuster toutes les images du dataset
    adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_used]

    # Comparer les images données avec les diff de 4 images même pose et 4 avec 1 image différente
    datatset4meme = adjusted_dataset[10:14]
    dataset1diff = adjusted_dataset[10:13] + [adjusted_dataset[-1]]

    diff_4meme, nb_4eme = process_4_images_pixel(datatset4meme, SEUIL_PIXEL_DIFF)
    diff_1diff, nb_1diff = process_4_images_pixel(dataset1diff, SEUIL_PIXEL_DIFF)
    print("Moy diff 4meme:", np.mean(diff_4meme))
    print("Moy diff 1diff:", np.mean(diff_1diff))

    # Afficher les images
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(diff_4meme, cmap='gray')
    plt.title("4 images même pose")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(nb_4eme, cmap='gray')
    plt.title("pixels diff > seuil (4meme)")
    plt.axis('off')
    plt.subplot(2, 2, 3)    
    plt.imshow(diff_1diff, cmap='gray')
    plt.title("3 images même pose + 1 différente")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(nb_1diff, cmap='gray')
    plt.title("pixels diff > seuil (1diff)")
    plt.axis('off')
    plt.show()



