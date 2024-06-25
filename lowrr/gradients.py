# Fonctions pour calculer les gradients

import numpy as np

def centered(img):
    """
    Calcule le gradient centré d'une image.
    
    :param img: Image en niveau de gris.
    :return: Gradient centré, même taille que l'image d'entrée.
             Le tableau retourné a une forme (nb_rows, nb_cols, 2)
             où la dernière dimension contient les composantes gx et gy.
    """
    # Vérifier que l'image a une taille suffisante pour calculer le gradient centré
    if img.ndim != 2:
        raise ValueError("L'image doit être une image 2D en niveau de gris")
    (rows, cols) = img.shape
    if rows < 3 or cols < 3:
        raise ValueError("L'image doit avoir au moins 3 pixels dans chaque direction")
    
    top = img[0:-2, 1:-1]
    bottom = img[2:, 1:-1]
    left = img[1:-1, 0:-2]
    right = img[1:-1, 2:]
    gx = np.zeros((rows, cols), dtype=np.int16)
    gy = np.zeros((rows, cols), dtype=np.int16)
    gx[1:-1, 1:-1] = right - left
    gy[1:-1, 1:-1] = bottom - top
    
    # Combiner les gradients dans une seule matrice
    gradient = np.stack([gx, gy], axis=-1)

    return gradient

# Calculer le carré de la norme du gradient directement à partir de l'image
def squared_norm_direct(img):
    # Vérifier que l'image a une taille suffisante pour calculer le gradient centré
    if img.ndim != 2:
        raise ValueError("L'image doit être une image 2D en niveau de gris")
    (rows, cols) = img.shape
    if rows < 3 or cols < 3:
        raise ValueError("L'image doit avoir au moins 3 pixels dans chaque direction")

    top = img[0:-2, 1:-1]
    bottom = img[2:, 1:-1]
    left = img[1:-1, 0:-2]
    right = img[1:-1, 2:]
    squared_norm_mat = np.zeros((rows, cols))
    squared_norm_mat[1:-1, 1:-1] = ((right - left)**2 + (bottom - top)**2)/4

    return squared_norm_mat
    