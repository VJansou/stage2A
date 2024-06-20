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
    if img.shape[0] < 3 or img.shape[1] < 3:
        raise ValueError("L'image doit avoir au moins 3 pixels dans chaque direction")
    
    # Déplacer les pixels de l'image pour calculer les gradients
    img_up = np.roll(img, -1, axis=0)
    img_down = np.roll(img, 1, axis=0)
    img_left = np.roll(img, -1, axis=1)
    img_right = np.roll(img, 1, axis=1)

    # Calculer les gradients
    gx = (img_right - img_left) / 2
    gy = (img_down - img_up) / 2

    # Combiner les gradients dans une seule matrice
    gradient = np.stack([gx, gy], axis=-1)

    return gradient