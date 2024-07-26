import cv2
import numpy as np
import matplotlib.pyplot as plt

from consts import *
import utils
import pyramide

def process_dataset(dataset):
    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]

    # Créer la pyramide d'images
    image_pyramid, levels = pyramide.mean_pyramid(MAX_LEVELS, dataset_gray)
    
    # Créer la matrice I pour chaque niveau
    I_matrices = {}
    I_trunc_matrices = {}
    diff_matrices = np.zeros(levels)
    for level in range(levels):
        I_matrices[level] = utils.create_I_matrix(image_pyramid[level])

        # Calculer la SVD de la matrice I
        U, S, Vt = np.linalg.svd(I_matrices[level], full_matrices=False)

        # Tronquer les valeurs singulières
        S_trunc = np.zeros_like(S)
        S_trunc[:1] = S[:1]

        # Reconstruire la matrice I avec les valeurs singulières tronquées
        I_trunc_matrices[level] = U @ np.diag(S_trunc) @ Vt

        # Déterminer l'écart entre les deux matrices
        diff_matrices[level] = np.linalg.norm(I_matrices[level] - I_trunc_matrices[level]) / np.linalg.norm(I_matrices[level])

    return diff_matrices

def main():
    datasets_paths = ["Dataset/test_00", "Dataset/test_01", "Dataset/test_02", "Dataset/test_10", "Dataset/test_11", "Dataset/test_12", "Dataset/test_20", "Dataset/test_21", "Dataset/test_22"]  # Ajoutez les chemins de tous vos datasets
    labels = datasets_paths  # Étiquettes pour chaque dataset

    plt.figure()
    
    for dataset_path, label in zip(datasets_paths, labels):
        # Charger les images
        dataset = utils.load_images(dataset_path)
        diff_matrices = process_dataset(dataset)

        # Courbe de la différence pour ce dataset
        plt.plot(diff_matrices, marker='o', linestyle='-', label=label)
    
    plt.xlabel('Niveau')
    plt.ylabel('Différence')
    plt.title('Différences entre les Matrices par Niveau de résolution')
    plt.xticks(range(len(diff_matrices)))  # Afficher les niveaux en tant que valeurs entières sur l'axe x
    plt.grid(True)
    plt.legend()
    plt.show()

main()
