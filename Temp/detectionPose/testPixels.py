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

    # # Vérifier que les dimensions sont correctes
    # print(I.shape)
    # print(I_trunc.shape)
    # if I_trunc.shape != I.shape:
    #     print("Erreur: les dimensions de la matrice I tronquée ne sont pas correctes.")
    #     return None

    # Déterminer l'écart pour chaque pixel
    diff = np.linalg.norm(I - I_trunc, axis=1)
    return diff, diff > seuil

if __name__ == "__main__":
    # Charger les images
    dataset = utils.load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)

    # Trie les images par date de prise de vue
    dataset_sorted = utils.sort_images_by_date(dataset)

    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset_sorted]

    # Recadrer les images
    dataset_cropped = [utils.crop_image(img, CROP_FACTOR) for img in dataset_gray]

    # Créer la pyramide d'images
    dataset_used = pyramideV2.last_mean_pyramid(NB_PIXELS_MIN, dataset_cropped)

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_used[0])

    # Ajuster toutes les images du dataset
    # adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_used]
    adjusted_dataset = utils.adjust_images_mean(dataset_used)

    # cv2.imshow("Image 1", adjusted_dataset[0])

    # Comparer les images données avec les diff de 4 images même pose et 4 avec 1 image différente
    datatset4meme = adjusted_dataset[:4]
    dataset1diff = adjusted_dataset[:3] + [adjusted_dataset[-1]]

    diff_4meme, nb_4eme = process_4_images_pixel(datatset4meme, SEUIL_PIXEL_DIFF)
    diff_1diff, nb_1diff = process_4_images_pixel(dataset1diff, SEUIL_PIXEL_DIFF)
    print("Moy diff 4meme:", np.mean(diff_4meme))
    print("Moy diff 1diff:", np.mean(diff_1diff))
    print("nb diff 4meme:", diff_4meme.size)

    # Calcul de la moyenne et de l'écart-type
    moy_4meme = np.mean(diff_4meme)
    std_4meme = np.std(diff_4meme)
    moy_1diff = np.mean(diff_1diff)
    std_1diff = np.std(diff_1diff)

    # # Enlever les valeurs aberrantes
    # diff_4meme = diff_4meme[diff_4meme < moy_4meme + 3 * std_4meme]
    # diff_1diff = diff_1diff[diff_1diff < moy_1diff + 3 * std_1diff]

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.hist(diff_4meme, bins=500, log=True)
    plt.xlabel("Différence de pixel")
    plt.ylabel("Nombre de pixels")
    plt.title("4 images même pose")
    plt.subplot(2, 2, 2)
    plt.hist(diff_1diff, bins=500, log=True)
    plt.xlabel("Différence de pixel")
    plt.ylabel("Nombre de pixels")
    plt.title("3 images même pose + 1 différente")
    plt.show()

    # Afficher les images
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(diff_4meme, cmap='gray')
    # plt.title("4 images même pose")
    # plt.axis('off')
    # plt.subplot(2, 2, 2)
    # plt.imshow(nb_4eme, cmap='gray')
    # plt.title("pixels diff > seuil (4meme)")
    # plt.axis('off')
    # plt.subplot(2, 2, 3)    
    # plt.imshow(diff_1diff, cmap='gray')
    # plt.title("3 images même pose + 1 différente")
    # plt.axis('off')
    # plt.subplot(2, 2, 4)
    # plt.imshow(nb_1diff, cmap='gray')
    # plt.title("pixels diff > seuil (1diff)")
    # plt.axis('off')
    # plt.show()



