# Fonctions utilitaires pour le projet
import os
import cv2
import numpy as np
from scipy import stats
from multiprocessing import Pool
import exifread
import sys
import contextlib
import illumination
from functools import partial

@contextlib.contextmanager
def suppress_all_output():
    """
    A context manager to suppress stdout and stderr.
    """
    new_stdout = os.devnull
    new_stderr = os.devnull
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(new_stdout, 'w') as devnull_out:
            with open(new_stderr, 'w') as devnull_err:
                sys.stdout = devnull_out
                sys.stderr = devnull_err
                yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def load_image(filepath):
    """
    Charge une image depuis le chemin spécifié.
    Renvoie l'image et la date de prise de vue si disponible.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Impossible de charger l'image", filepath)
    
    # Lire les métadonnées de l'image
    with open(filepath, 'rb') as f:
        try:
            with suppress_all_output():
                tags = exifread.process_file(f)
            date_time_taken = tags.get('EXIF DateTimeOriginal')
        except:
            date_time_taken = None
    
    return img, date_time_taken

def load_images(directory, num_workers=4):
    """
    Charge toutes les images d'un répertoire dans une liste en utilisant le parallélisme.
    Renvoie une liste de tuples (image, date de prise de vue).
    """
    filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    with Pool(num_workers) as p:
        images_metadata = p.map(load_image, filepaths)

    return images_metadata

def sort_images_by_date(images_metadata):
    """
    Trie les images par date de prise de vue.
    Si une des images n'a pas de date de prise de vue, affiche un message et ne trie pas.
    Renvoie seulement les images triées. (pas les dates de prise de vue)
    """
    if any([datetime_taken is None for _, datetime_taken in images_metadata]):
        print("Certaines images n'ont pas de date de prise de vue. Impossible de trier.")
        return [img for img, _ in images_metadata]   
     
    sorted_images_metadata = sorted(images_metadata, key=lambda x: x[1].values)
    return [img for img, _ in sorted_images_metadata]


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

def adjust_image_mean_V2(img, H_ref):
    """
    Ajuste l'histogramme d'une image pour correspondre à un histogramme cumulé de référence.
    """
    H = illumination.calculate_cumulative_histogram(img)
    f = illumination.create_transformation_fonction(H, H_ref)
    return f[img]

def adjust_images_mean(images, num_workers=4):
    """
    Ajuste le niveau de gris de toutes les images pour correspondre à l'histogramme cumulé de la première image.
    """
    H_ref = illumination.calculate_cumulative_histogram(images[0])
    with Pool(num_workers) as p:
        dataset_adjusted = p.map(partial(adjust_image_mean_V2, H_ref=H_ref), images[1:])
    return dataset_adjusted

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