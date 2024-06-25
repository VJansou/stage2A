# Fonctions pour générer des images multirésolution

import numpy as np
import cv2
import matplotlib.pyplot as plt

def mean_pyramid(max_levels, dataset):
    """
    Génère une pyramide d'images multirésolution à partir d'un dataset d'images.

    :param dataset: Liste des images originales.
    :param max_levels: Nombre de niveaux max dans la pyramide.
    :return: Liste de listes d'images, une pour chaque niveau de la pyramide et le nombre de niveaux effectivement générés.
    """
    # Initialiser les images multirésolution
    images = dataset.copy()

    # Générer les images multirésolution
    tmp = []
    for im in images:
        tmp.append(limited_sequence(max_levels, im, lambda x: halve(x, lambda a, b, c, d: (a/4 + b/4 + c/4 + d/4))))
    
    # Changer la structure de la pyramide
    image_pyramid = []
    for level in range(len(tmp[0])):
        image_pyramid.append([tmp[img][level] for img in range(len(tmp))])
    
    return image_pyramid, len(image_pyramid)


def limited_sequence(max_length, data, f):
    """
    Appel récursif à une fonction qui transforme une image
    jusqu'à ce que la longueur de la séquence atteigne un maximum ou 
    que ce ne soit plus possible.
    """
    length = 1
    def f_limited(data):
        nonlocal length
        if length < max_length:
            length += 1
            return f(data)
        else:
            return None
    return sequence(data, f_limited)


def sequence(data, f):
    """
    Appel récursif à une fonction qui transforme une image
    jusqu'à ce que ce ne soit plus possible.
    """
    s = [data]
    while True:
        new_data = f(s[-1])
        if new_data is None:
            break
        s.append(new_data)
    return s

def halve(mat, f):
    """
    Réduit la taille d'une matrice de moitié en appliquant une fonction à chauqe bloc 2x2.

    Si une ligne ou une colonne a une taille < 2, renvoie None.
    Si une ligne ou une colonne a une taille impaire, la dernière ligne/colonne est enlevée.
    """
    mat = np.asarray(mat)
    (rows, cols) = mat.shape
    if rows < 2 or cols < 2:
        return None
    half_rows = rows // 2
    half_cols = cols // 2
    half_mat = np.zeros((half_rows, half_cols), dtype=mat.dtype)
    for i in range(half_rows):
        for j in range(half_cols):
            a = mat[2 * i, 2 * j]
            b = mat[2 * i + 1, 2 * j]
            c = mat[2 * i, 2 * j + 1]
            d = mat[2 * i + 1, 2 * j + 1] 
            half_mat[i, j] = f(a, b, c, d)
    return half_mat


# Fonction pour montrer la première image de la pyramide

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
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images_to_show):
        plt.subplot(1, num_levels, i + 1)
        # Convertir l'image en RGB si elle est en BGR (OpenCV charge en BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"Résolution niveau {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()