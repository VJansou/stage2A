# Fonction pour générer des images multirésolution

import cv2
import matplotlib.pyplot as plt

def run(dataset, levels):
    """
    Génère une pyramide d'images multirésolution à partir d'un dataset d'images.

    :param dataset: Liste des images originales.
    :param levels: Nombre de niveaux dans la pyramide.
    :return: Liste de listes d'images, une pour chaque niveau de la pyramide.
    """
    # Initialiser les images multirésolution
    images = [dataset.copy()]

    # Générer les images multirésolution
    for _ in range(1, levels):
        # Réduire la taille des images
        reduced_images = [cv2.pyrDown(img) for img in images[-1]]
        print(reduced_images[0].dtype)
        images.append(reduced_images)

    return images


# Fonctions pour tester run

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

# def test():
#     # Charge les images
#     dataset = m.load_images(m.IN_DIR)
#     # Convertit les images en niveaux de gris
#     dataset = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
#     # Génère les images multirésolution
#     image_pyramid = run(dataset, m.LEVELS)
#     # Affiche les dimensions des images à chaque niveau
#     for level, images_at_level in enumerate(image_pyramid):
#         print(f"Niveau {level}:")
#         for img_idx, img in enumerate(images_at_level):
#             print(f"  Image {img_idx}: Dimensions {img.shape}")


# test()