# Fonctions pour faire un prétraitement des images

import numpy as np

def illumination_correction(img1, img2, threshold=0.1):
    """
    Corrige l'illumination des images en les ajustant pour qu'elles aient la même moyenne.
    """
    if img1.shape != img2.shape:
        raise ValueError("Les images doivent être de la même taille.")
    
    # Calculer l'intensité moyenne de chaque image
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    # Vérifier si la correction est nécessaire
    if np.abs(mean1 - mean2) < threshold:
        print("Aucune correction nécessaire.")
        return img1, img2

    # Calculer l'histogramme cumulé pour chaque image
    H1 = calculate_cumulative_histogram(img1)
    H2 = calculate_cumulative_histogram(img2)

    # Calculer le nouvel histogramme cumulé
    H_new = (H1 + H2) // 2

    # Calculer la fonction de transformation pour chaque image
    f1 = create_transformation_fonction(H1, H_new)
    f2 = create_transformation_fonction(H2, H_new)

    # Appliquer la transformation à chaque image
    img1_corrected = f1[img1]
    img2_corrected = f2[img2]

    return img1_corrected, img2_corrected


def calculate_cumulative_histogram(img):
    """
    Calcule l'histogramme cumulé d'une image.
    """
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    return np.cumsum(hist)


def create_transformation_fonction(H, H_new):
    """
    Crée une fonction de transformation pour ajuster l'illumination d'une image.
    """
    return np.argmin(np.abs(H[:, np.newaxis] - H_new), axis=1).astype(np.uint8)
