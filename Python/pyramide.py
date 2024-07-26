# Fonctions pour créer une pyramide d'images

import numpy as np
import matplotlib.pyplot as plt

def mean_pyramid(max_levels, dataset):
    """
    Génère une pyramide d'images multirésolution à partir d'un dataset d'images.

    :param dataset: Liste des images originales (tableaux NumPy 2D et toutes de même dimension).
    :param max_levels: Nombre de niveaux max dans la pyramide.
    :return: Liste de listes d'images, une pour chaque niveau de la pyramide et le nombre de niveaux effectivement générés.
    """
    # Initialiser les images multirésolution
    pyramids = [create_pyramid(im, max_levels) for im in dataset]
    
    # Déterminer le nombre de niveaux effectivement générés
    levels = len(pyramids[0])

    # Organiser la pyramide par niveaux
    image_pyramid = [np.array([pyramid[level] for pyramid in pyramids]) for level in range(levels)]
    
    return image_pyramid, levels

def create_pyramid(image, max_levels):
    """
    Génère les niveaux de la pyramide pour une seule image.

    :param image: Image originale (tableau NumPy 2D).
    :param max_levels: Nombre de niveaux max dans la pyramide.
    :return: Liste des images pour chaque niveau de la pyramide.
    """
    return limited_sequence(max_levels, image, lambda x: halve(x, lambda a, b, c, d: np.uint8(np.rint(a / 4 + b / 4 + c / 4 + d / 4))))


def limited_sequence(max_length, data, f):
    """
    Appel récursif à une fonction qui transforme une image
    jusqu'à ce que la longueur de la séquence atteigne un maximum ou 
    que ce ne soit plus possible.
    """
    sequence = [data]
    for _ in range(max_length - 1):
        new_data = f(sequence[-1])
        if new_data is None:
            break
        sequence.append(new_data)
    return sequence


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


def show_first_image(image_pyramid):
    """
    Affiche la première image du dataset à différentes résolutions mais à la même taille.

    :param image_pyramid: Liste de listes d'images, une pour chaque niveau de la pyramide.
    """
    # Extraire la première image à chaque niveau de la pyramide
    images_to_show = [level[0] for level in image_pyramid]

    # Déterminer le nombre de niveaux
    num_levels = len(images_to_show)

    # Créer une figure avec une sous-figure pour chaque niveau
    plt.figure()
    for i, img in enumerate(images_to_show):
        plt.subplot(1, num_levels, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Résolution niveau {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()