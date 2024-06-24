import numpy as np

# Fonctions pour manipuler le paramètre de mouvement

# Créer une matrice de projection à partir des paramètres de mouvement
def projection_mat(params):
    return np.array([
        [1 + params[0], params[2], params[4]],
        [params[1], 1 + params[3], params[5]],
        [0, 0, 1]
    ])

# Créer un vecteur de paramètres de mouvement à partir d'une matrice de projection
def projection_params(mat):
    return np.array([mat[0,0] - 1, mat[1,0], mat[0,1], mat[1,1] - 1, mat[0,2], mat[1,2]])