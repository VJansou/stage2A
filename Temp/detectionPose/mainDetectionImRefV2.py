import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from consts import *
import utils
import pyramide

def process_4_images(dataset):
    # Vérifier que le dataset contient 4 images
    if len(dataset) != 4:
        print("Erreur: le dataset doit contenir 4 images.")
        return None

    # Créer la pyramide d'images
    image_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, dataset)

    # Créer la matrice I pour le niveau 4
    I = utils.create_I_matrix(image_pyramid[4])

    # Calculer la SVD de la matrice I
    U, S, Vt = np.linalg.svd(I, full_matrices=False)

    # Tronquer les valeurs singulières
    S_trunc = np.zeros_like(S)
    S_trunc[:3] = S[:3]

    # Reconstruire la matrice I avec les valeurs singulières tronquées
    I_trunc = U @ np.diag(S_trunc) @ Vt

    # Déterminer l'écart entre les deux matrices
    diff = np.linalg.norm(I - I_trunc)
    return diff

def main():
    """
    On considère :
        qu'on a au moins 3 images pour une même pose.
        qu'on peux aller au niveau 4 de la pyramide.
    """
    # Charger les images
    dataset = utils.load_images(IN_DIR)
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    nb_images = len(dataset)
    print(nb_images, "images chargées.")

    # Calculer la moyenne des niveaux de gris de l'image de référence
    mean_reference = np.mean(dataset_gray[0])

    # Ajuster toutes les images du dataset
    adjusted_dataset = [utils.adjust_image_mean(img, mean_reference) for img in dataset_gray]

    # Vérifier qu'on a au moins 4 images
    if nb_images < 4:
        print("Erreur: le dataset doit contenir au moins 4 images.")
        return
    
    # Boucle principale
    continue_processing = True
    num_img_ref = 0
    diff_values = np.zeros((nb_images-2, nb_images))  # Matrice pour stocker les valeurs de diff
    while continue_processing:
        # Sélectionner 1 image si possible
        if num_img_ref < nb_images-2:
            images_ref = adjusted_dataset[num_img_ref:num_img_ref+3]

            # Calucler la différence entre les images
            for num_image in range(nb_images):
                if num_image == num_img_ref or num_image == num_img_ref+1 or num_image == num_img_ref+2:
                    diff_values[num_img_ref, num_image] = np.nan
                else:
                    diff_values[num_img_ref, num_image] = (process_4_images(images_ref + [adjusted_dataset[num_image]]))
            num_img_ref += 1
        else:
            continue_processing = False
    
    # # Afficher les différences
    # print("Valeurs de diff: ", diff_values)
    # plt.figure()
    # plt.plot(diff_values, 'o-')
    # plt.title('Différences')
    # plt.xlabel('Index')
    # plt.ylabel('Différence')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.show()

    # Afficher les différences sous forme de heatmap
    # Déterminer les limites pour la barre de couleur
    # vmin = np.nanmin(diff_values)
    # vmax = np.nanmax(diff_values)
    # print(vmin, vmax)

    # plt.figure(figsize=(12, 6))
    # sns.heatmap(diff_values, cmap='viridis', cbar_kws={'label': 'Valeurs de différence'}, vmin=vmin, vmax=vmax)
    # plt.title('Différences entre les images')
    # plt.xlabel('Numéro de l\'image')
    # plt.ylabel('Index de référence')
    # plt.show()

    # Tracer les différences avec vmin et vmax adaptés par ligne
    fig, axs = plt.subplots(nrows=nb_images-2, figsize=(15, 2 * (nb_images-2)), constrained_layout=True)

    for i in range(nb_images-2):
        # Déterminer les limites pour la barre de couleur pour chaque ligne
        vmin = np.nanmin(diff_values[i, :])
        vmax = np.nanmax(diff_values[i, :])
        
        # Créer une heatmap pour chaque ligne
        sns.heatmap([diff_values[i, :]], cmap='viridis', xticklabels=False, yticklabels=False,
                    vmin=vmin, vmax=vmax, ax=axs[i])

    plt.show()
main()
                


