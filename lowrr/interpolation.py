# Fonctions pour interpoler/extrapoler les images

import numpy as np

#  Interpolation linéaire simple d'un pixel avec des coordonnées en virgule flottante.
#  Extrapole avec la bordure la plus proche si le point est en dehors des limites de l'image.
def linear(x, y, image):
    (height, width) = image.shape
    u = int(np.floor(x))
    v = int(np.floor(y))
    if u >= 0 and u < width - 2 and v >= 0 and v < height - 2:
        u_0 = u
        u_1 = u + 1
        v_0 = v
        v_1 = v + 1
        a = x - u
        b = y - v

        vu_00 = image[v_0, u_0]
        vu_01 = image[v_0, u_1]
        vu_10 = image[v_1, u_0]
        vu_11 = image[v_1, u_1]

        interp = (1 - a) * (1 - b) * vu_00
        interp += a * (1 - b) * vu_01
        interp += (1 - a) * b * vu_10
        interp += a * b * vu_11
        return interp
    else:
        # Extrapolation du voisin le plus proche à l'extérieur des frontières
        nearest_v, nearest_u = nearest_border(x, y, width, height)
        return image[nearest_v, nearest_u]
    
def nearest_border(x, y, width, height):
    u = min(max(x, 0), width - 1)
    v = min(max(y, 0), height - 1)
    return (int(v), int(u))