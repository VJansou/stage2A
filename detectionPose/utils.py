# Fonctions utilitaires pour le projet
import os
import cv2
import numpy as np
from scipy import stats
from multiprocessing import Pool

def load_image(filepath):
    """
    Charge une image depuis le chemin spécifié.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Impossible de charger l'image", filepath)
    else:
        return img

def load_images(directory, num_workers=4):
    """
    Charge toutes les images d'un répertoire dans une liste en utilisant le parallélisme.
    """
    filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    with Pool(num_workers) as p:
        images = p.map(load_image, filepaths)

    return images


def convert_to_gray_single(img):
    """
    Convertit une image en niveaux de gris.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convert_to_gray(dataset, num_workers=4):
    """
    Convertit une liste d'images en niveaux de gris en utilisant le parallélisme.
    """
    with Pool(num_workers) as p:
        dataset_gray = p.map(convert_to_gray_single, dataset)

    return dataset_gray

# def load_images(directory):
#     """
#     Charge toutes les images d'un répertoire dans une liste.
#     """
#     images = []
#     for filename in os.listdir(directory):
#         img = cv2.imread(os.path.join(directory, filename))
#         if img is not None:
#             images.append(img)
#         else:
#             print("Erreur: n'a pas trouvé l'image", filename)
#     return images

def create_I_matrix(images):
    """
    Crée la matrice I pour une liste d'images de même dimension.
    """
    return np.vstack([image.flatten() for image in images]).transpose()

def adjust_image_mean(img, target_mean):
    """
    Ajuste le niveau de gris d'une image pour correspondre à une moyenne cible.
    """
    current_mean = np.mean(img)
    adjusted_img = img + (target_mean - current_mean)
    # S'assurer que les valeurs restent dans l'intervalle [0, 255]
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    return adjusted_img

def crop_image(image, factor):
    """
    Recadre une image en conservant le centre.
    input:
        image: image à recadrer
        factor: facteur de recadrage (0 < factor <= 1)
    """
    if factor <= 0 or factor > 1:
        raise ValueError("Le facteur de recadrage doit être compris entre 0 et 1.")
    
    height, width = image.shape
    new_height, new_width = int(height * factor), int(width * factor)
    
    start_h = (height - new_height) // 2
    start_w = (width - new_width) // 2
    end_h = start_h + new_height
    end_w = start_w + new_width
    return image[start_h:end_h, start_w:end_w]

def grubbs_test_first_value_inf(data, alpha=0.05):
    """
    Test de Grubbs pour détecter si la première 
    valeur est nettement inférieure aux autres dans une liste de valeurs.
    Renvoie toujours False si la liste contient moins de 6 valeurs.
    """
    n = len(data)
    if n < 6:
        return False

    # Exclure la première valeur
    data_without_first = data[1:]

    mean = np.mean(data_without_first)
    std_dev = np.std(data_without_first)

    # Calculer le score de Grubbs pour la première valeur
    G_first = np.abs(data[0] - mean) / std_dev

    # Calculer la statistique de Grubbs
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    critical_value = (n - 1) / np.sqrt(n) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

    return G_first > critical_value