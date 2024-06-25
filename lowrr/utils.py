# Fonctions utilitaires pour le projet LowRR

import sparse

# Donne un itérateur pour les coordonnées en colonne
def coords_col_major(shape):
    (rows, cols) = shape
    return ((x, y) for x in range(cols) for y in range(rows))

# Donne les coordonnées des pixels épars
def coordinates_from_mask(mask):
    return sparse.extract(mask, coords_col_major(mask.shape))