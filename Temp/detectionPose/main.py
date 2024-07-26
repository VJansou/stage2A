import cv2
import numpy as np
import matplotlib.pyplot as plt

from consts import *
import utils
import pyramide

def main():
    # Charger les images
    dataset = utils.load_images(IN_DIR)
    nb_images = len(dataset)
    if DEBUG:
        print(nb_images, "images chargées.")
        # Affiche la première image
        cv2.imshow("Premiere image", dataset[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    if DEBUG:
        # Affiche la première image
        cv2.imshow("Premiere image en niveau de gris", dataset_gray[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Créer la pyramide d'images
    image_pyramid, levels = pyramide.mean_pyramid(MAX_LEVELS, dataset_gray)
    if DEBUG:
        # Affiche la première image, ses dimensions et son type à chaque niveau
        for level, lvl_imgs in enumerate(image_pyramid):
            print(f"Niveau {level}:")
            print(f"  Image 0: --> Dimensions {lvl_imgs[0].shape}")
            print(f"           --> Type {lvl_imgs[0].dtype}")
        pyramide.show_first_image(image_pyramid)
    
    # Créer la matrice I pour chaque niveau
    I_matrices = {}
    I_trunc_matrices = {}
    diff_matrices = np.zeros(levels)
    for level in range(levels):
        I_matrices[level] = utils.create_I_matrix(image_pyramid[level])

        # Calculer la SVD de la matrice I
        U, S, Vt = np.linalg.svd(I_matrices[level], full_matrices=False)
        if DEBUG:
            print(f"\nvaleurs singulières niveau {level}: {S}")
        S_trunc = np.zeros_like(S)
        S_trunc[:1] = S[:1]
        if DEBUG:
            print(f"valeurs singulières tronquées niveau {level}: {S_trunc}")
            print(f"valeur singulières enlevées niveau {level}: {S[1:]}")

        # Reconstruire la matrice I avec les valeurs singulières tronquées
        I_trunc_matrices[level] = U @ np.diag(S_trunc) @ Vt
        if DEBUG:
            print(f"Shape I niveau {level}: {I_matrices[level].shape}")
            print(f"Shape I_trunc niveau {level}: {I_trunc_matrices[level].shape}")
    
        # Déterminer l'écart entre les deux matrices
        diff_matrices[level] = np.linalg.norm(I_matrices[level] - I_trunc_matrices[level]) / np.linalg.norm(I_matrices[level])

        # Afficher les différences
        print(f"\nDifférence niveau {level}: {diff_matrices[level]:.4f}")

    # Courbe de la différence
    plt.figure()
    plt.plot(diff_matrices, marker='o', linestyle='-', color='b', label='Différence')
    plt.xlabel('Niveau')
    plt.ylabel('Différence')
    plt.title('Différences entre les Matrices par Niveau')
    plt.xticks(range(len(diff_matrices)))  # Afficher les niveaux en tant que valeurs entières sur l'axe x
    plt.grid(True)
    plt.legend()
    plt.show()

main()