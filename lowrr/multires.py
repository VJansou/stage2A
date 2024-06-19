# Fonction pour générer des images multirésolution

import cv2

def run(dataset, levels):
    # Initialiser les images multirésolution
    images = [dataset.copy()]

    # Générer les images multirésolution
    for _ in range(1, levels):
        # Réduire la taille des images
        reduced_images = [cv2.pyrDown(img) for img in images[-1]]
        images.append(reduced_images)

    return images


# Test de la fonction multires
# import main as m
# def main():
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


# main()