import cv2
import numpy as np
import matplotlib.pyplot as plt

from consts import *
import utils
import pyramide

def process_4_images(dataset, diff_values):
    # Vérifier que le dataset contient 4 images
    if len(dataset) != 4:
        print("Erreur: le dataset doit contenir 4 images.")
        return None
    
    # Convertir les images en niveaux de gris
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]

    # Créer la pyramide d'images
    image_pyramid, _ = pyramide.mean_pyramid(MAX_LEVELS, dataset_gray)

    # Créer la matrice I pour le niveau 3
    I3 = (image_pyramid[3][0] + image_pyramid[3][1] + image_pyramid[3][2])/3
    I4 = (I3 + image_pyramid[3][3])/2

    # Déterminer si l'écart entre les deux matrices dépasse le seuil
    diff = np.linalg.norm(I3 - I4) / np.linalg.norm(I3)
    diff_values.append(diff)  # Ajouter la valeur de diff à la liste globale
    return diff > SEUIL

def main():
    """
    On considère :
        qu'on a au moins 3 images pour une même pose.
        qu'on peux aller au niveau 3 de la pyramide.
    """
    # Charger les images
    dataset = utils.load_images(IN_DIR)
    nb_images = len(dataset)
    print(nb_images, "images chargées.")

    # Vérifier qu'on a au moins 3 images
    if nb_images < 3:
        print("Erreur: le dataset doit contenir au moins 3 images.")
        return
    
    # Boucle principale
    continue_processing = True
    num_changement_pose = []
    diff_values = []  # Liste pour stocker les valeurs de diff
    start_indices = []  # Liste pour stocker les indices de début des groupes de 4 images
    num_image = 0
    while continue_processing:
        # Sélectionner 4 images si possible
        if num_image + 4 <= nb_images:
            start_indices.append(num_image)
            images = dataset[num_image:num_image + 4]

            # Vérifier si les 4 images correspondent à la même pose
            if process_4_images(images, diff_values):
                num_changement_pose.append(num_image + 3)
                num_image += 4
            else:
                num_image += 1
        else:
            continue_processing = False
    
    # Afficher les indices des images où il y a un changement de pose
    print("Indices des images où il y a un changement de pose:", num_changement_pose)

    # Tracer les valeurs de diff
    plt.figure()
    plt.plot(start_indices, diff_values, marker='o', linestyle='-', color='b', label='Différence')
    plt.xlabel('Groupes de 4 images')
    plt.ylabel('Différence')
    plt.title('Valeurs de différence pour chaque groupe de 4 images')
    plt.grid(True)
    plt.legend()
    plt.show()

main()
                


