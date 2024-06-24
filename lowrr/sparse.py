# Sélection de points épars de manière grossière à fine.
# Utile pour accélérer les algorithmes de registration.

import numpy as np

# Sélectionne un sous-ensemble de points satisfaisant deux conditions :
#   * les points doivent être bien répartis dans l'image.
#   * une densité plus élevée là où les gradients sont plus grands.

# Chaque niveau est conservé, mais le plus important
# est celui à la plus haute résolution (le dernier).

def select(diff_threshold, gradients):
    (rows, cols) = gradients[-1].shape
    init_sparse = np.full((rows, cols), True, dtype=bool)
    prune = lambda a, b, c, d: prune_with_thresh(diff_threshold, a, b, c, d)

    # On commence par la résolution la plus basse
    # On saute la première résolution car tout les points sont bons
    for grad_mat in reversed(gradients[1:]):
        new_mask = select_2x2_bloc(init_sparse[-1], grad_mat, prune)
        init_sparse.append(new_mask)
    return init_sparse


# Applique une fonction prédicat sur chaque bloc 2x2 de mat
# Evalue seulement les blocs dans le masque pre_mask
def select_2x2_bloc(pre_mask, mat, f):
    (rows, cols) = mat.shape
    (rows2, cols2) = pre_mask.shape
    assert rows == rows2 * 2 and cols == cols2 * 2
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