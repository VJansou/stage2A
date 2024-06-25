# Sélection de points épars de manière grossière à fine.
# Utile pour accélérer les algorithmes de registration.

import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import ceil

# Sélectionne un sous-ensemble de points satisfaisant deux conditions :
#   * les points doivent être bien répartis dans l'image.
#   * une densité plus élevée là où les gradients sont plus grands.

# Chaque niveau est conservé, mais le plus important
# est celui à la plus haute résolution (le dernier).

def select(diff_threshold, gradients):
    (rows, cols) = gradients[-1].shape
    # print(f"rows: {rows}, cols: {cols}")
    # for level in gradients:
    #     print(f"level shape: {level.shape}")
    init_sparse = np.full((rows, cols), True, dtype=bool)
    prune = lambda a, b, c, d: prune_with_thresh(diff_threshold, a, b, c, d)
    multires_mask = []
    multires_mask.append(init_sparse)
    # On commence par la résolution la plus basse
    # On saute la première résolution car tout les points sont bons
    for grad_mat in reversed(gradients[:-1]):
        new_mask = select_2x2_bloc(multires_mask[-1], grad_mat, prune)
        multires_mask.append(new_mask)
    return multires_mask[::-1]


# Applique une fonction prédicat sur chaque bloc 2x2 de mat
# Evalue seulement les blocs dans le masque pre_mask
def select_2x2_bloc(pre_mask, mat, f):
    (rows, cols) = mat.shape
    (rows2, cols2) = pre_mask.shape
    assert ceil(rows / 2) == rows2 and ceil(cols / 2) == cols2, f"Assertion failed: rows / 2 ({ceil(rows / 2)}) != rows2 ({rows2}) or cols / 2 ({ceil(cols / 2)}) != cols2 ({cols2})"
    mask = np.zeros((rows, cols), dtype=bool)
    for j in range(cols2):
        for i in range(rows2):
            if pre_mask[i, j]:
                a = mat[2*i, 2*j]
                b = mat[2*i+1, 2*j]
                c = mat[2*i, 2*j+1]
                d = mat[2*i+1, 2*j+1]
                ok = f(a, b, c, d)
                mask[2*i, 2*j] = ok[0]
                mask[2*i+1, 2*j] = ok[1]
                mask[2*i, 2*j+1] = ok[2]
                mask[2*i+1, 2*j+1] = ok[3]
    return mask


# Enlève les 2 ou 3 valeurs les plus faibles.
# La deuxième valeur la plus élévée doit être plus grande que la troisième + thresh:
#     second > third + thresh

# Par exemple: avec thresh = 5
#     ( 0, 1, 8, 9 ) -> [ false, false, true, true ]
#     ( 0, 9, 1, 8 ) -> [ false, true, false, true ]
#     ( 1, 0, 9, 0 ) -> [ false, false, true, false ]
def prune_with_thresh(tresh, a, b, c, d):
    l = [a, b, c, d]
    l_sorted = sorted(l)
    third = l_sorted[-3]
    second = l_sorted[-2]
    first = l_sorted[-1]
    result = [(x == second and second > third + tresh) for x in l]
    result[l.index(first)] = True
    return result

# Tests
print(prune_with_thresh(5, 0, 1, 8, 9)) # [False, False, True, True]
print(prune_with_thresh(5, 0, 9, 1, 8)) # [False, True, False, True]
print(prune_with_thresh(5, 1, 0, 9, 0)) # [False, False, True, False]

# Pour tester select
def show_first_image(image_pyramid, multires_sparse_pixels):

    # Extraire la première image et son masque à chaque niveau de la pyramide
    images_to_show = [level[0] for level in image_pyramid]
    masks_to_show = [level[0] for level in multires_sparse_pixels]

    # Déterminer le nombre de niveaux
    num_levels = len(images_to_show)

    # Créer une figure avec une sous-figure pour chaque niveau
    plt.figure(figsize=(15, 5))
    for i, (img, mask) in enumerate(zip(images_to_show, masks_to_show)):
        plt.subplot(1, num_levels, i + 1)
        # Convertir l'image en RGB si elle est en BGR (OpenCV charge en BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.imshow(mask, cmap='hot', alpha=0.5)
        plt.title(f"Résolution niveau {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

        

