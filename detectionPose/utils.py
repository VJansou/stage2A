# Fonctions utilitaires pour le projet
import os
import cv2
import numpy as np

def load_images(directory):
    """
    Charge toutes les images d'un répertoire dans une liste.
    """
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
        else:
            print("Erreur: n'a pas trouvé l'image", filename)
    return images

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