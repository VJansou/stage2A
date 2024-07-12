# Idée : le nombre de level sera déterminé en fonction du 
# nombre de pixel qu'on veut avoir dans l'image finale.

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

def last_mean_pyramid(nb_pixel_min, dataset, num_workers=4):
    """
    Renvoie le plus haut niveau de la pyramide qui a un nombre de pixels
    par ligne/colonne supérieur ou égal à nb_pixel_min.
    """
    # Déterminer le nombre de niveau dans la pyramide
    levels = max(0, np.floor(np.log2(min(dataset[0].shape) / nb_pixel_min)).astype(int) + 1)
    print("Nombre de niveaux dans la pyramide:", levels)

    # Créer le dernier niveau de la pyramide
    with multiprocessing.Pool(num_workers) as p:
        last_images_pyramid = p.map(partial(last_create_pyramid, levels=levels), dataset)
    return last_images_pyramid


def last_create_pyramid(image, levels):
    """
    Renvoie le plus haut niveau de la pyramide pour une seule image.

    :param image: Image originale (tableau NumPy 2D).
    :param level: Nombre de niveaux dans la pyramide.
    """
    return limited_sequence(levels, image, lambda x: halve(x, mean_block))


def limited_sequence(max_length, data, f):
    """
    Appel récursif à une fonction qui transforme une image
    jusqu'à ce que la longueur de la séquence atteigne un maximum ou 
    que ce ne soit plus possible.
    """
    sequence = data
    for _ in range(max_length - 1):
        new_data = f(sequence)
        if new_data is None:
            break
        sequence = new_data
    return sequence


def mean_block(a, b, c, d):
    """
    Calcule la moyenne des 4 blocs 2x2.
    """
    return np.uint8(np.rint(a / 4 + b / 4 + c / 4 + d / 4))


def halve(mat, f):
    """
    Réduit la taille d'une matrice de moitié en appliquant une fonction à chaque bloc 2x2.

    Si une ligne ou une colonne a une taille < 2, renvoie None.
    Si une ligne ou une colonne a une taille impaire, la dernière ligne/colonne est enlevée.
    """
    mat = np.asarray(mat, dtype=mat.dtype)
    (rows, cols) = mat.shape
    if rows < 2 or cols < 2:
        return None
    
    # Ajustement pour les dimensions impaires
    if rows % 2 == 1:
        mat = mat[:-1, :]
    if cols % 2 == 1:
        mat = mat[:, :-1]
    
    # Extraction des blocs 2x2
    mat_00 = mat[0::2, 0::2]
    mat_01 = mat[0::2, 1::2]
    mat_10 = mat[1::2, 0::2]
    mat_11 = mat[1::2, 1::2]

    # Application de la fonction f
    return f(mat_00, mat_01, mat_10, mat_11)
