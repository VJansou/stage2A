import cv2
import os
import numpy as np

from consts import *
import multires
import gradients
import interpolation

# Charge les images d'un répertoire
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
        else:
            print("Erreur: n'a pas trouvé l'image", filename)
    return images

# Réduit les valeurs vers 0
def shrink(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

# Donne un itérateur pour les coordonnées en colonne
def coords_col_major(shape):
    (rows, cols) = shape
    return ((x, y) for x in range(cols) for y in range(rows))

# Calcule le pas de Gauss-Newton
def forwards_compositional_step(shape, coordinates, residuals, gradient):
    # Initialisation des paramètres de descente
    (height, width) = shape
    descent_params = np.zeros(6)
    hessian = np.zeros((6,6))
    border = int(0.04 * min(height, width))
    pixel_count_inside = 0

    for (((x, y), res), (gx, gy)) in zip(coordinates, residuals, gradient):
        # Vérifie si le pixel est dans la région d'intérêt
        if x > border and x < width - border and y > border and y < height - border:
            pixel_count_inside += 1

            # Calcul de l'approximation de Gauss-Newton
            jacobian = np.array([x*gx, x*gy, y*gx, y*gy, gx, gy])
            hessian += np.outer(jacobian, jacobian)

            # Calcul de la descente
            descent_params += res * jacobian
    if pixel_count_inside < 6:
        raise ValueError("Pas assez de pixels dans la région d'intérêt")
    hessian_chol = np.linalg.cho_factor(hessian)
    return np.linalg.cho_solve(hessian_chol, descent_params)

# Crée une matrice de projection à partir des paramètres de mouvement
def projection_mat(params):
    return np.array([[1 + params[0], params[1], params[2]],
                     [params[3], 1 + params[4], params[5]],
                     [0, 0, 1]])

# Crée un vecteur de paramètres de mouvement à partir d'une matrice de projection
def projection_params(mat):
    return np.array([mat[0,0] - 1, mat[0,1], mat[0,2], mat[1,0], mat[1,1] - 1, mat[1,2]])

# Projette les coordonnées des pixels sur l'image
def project(coordinates, registered, imgs, motion_vector):
    for (i, motion) in enumerate(motion_vector):
        motion_mat = projection_mat(motion)
        registered_col = registered[:,i]
        for (x, y), pixel in zip(coordinates, range(len(registered_col))):
            new_pos = np.dot(motion_mat, np.array([x, y, 1.0]))
            interp = interpolation.linear(new_pos[0], new_pos[1], imgs[i])
            registered_col[pixel] = interp

# Fonction principale
def main():
    # Charge les images
    dataset = load_images(IN_DIR)
    nb_images = len(dataset)
    if DEBUG:
        # Affiche la première image
        cv2.imshow("Premiere image", dataset[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convertit les images en niveaux de gris
    dataset = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    if DEBUG:
        # Affiche la première image
        cv2.imshow("Premiere image en niveau de gris", dataset[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Génère les images multirésolution
    image_pyramid = multires.run(dataset, LEVELS)
    if DEBUG:
        # Affiche les dimensions de la première image à chaque niveau
        for level, images_at_level in enumerate(image_pyramid):
            print(f"Niveau {level}:")
            print(f"  Image 0: Dimensions {images_at_level[0].shape}")
        multires.show_first_image(image_pyramid)
    
    # Initialisation du vecteur de mouvement
    motion_vector = np.zeros([nb_images,6])

    # Algorithme multi-résolution.
    # Effectue la même opération à chaque niveau pour les images et gradients correspondants.
    # L'itérateur est inversé pour commencer par le dernier niveau (la résolution la plus basse).
    # Le niveau 0 correspond aux images initiales.
    image_pyramid_inv = image_pyramid[::-1]

    for level, images_at_level in enumerate(image_pyramid_inv):
        (rows, cols) = images_at_level[0].shape

        # Adaptation du vecteur de mouvement au changement de résolution
        motion_vector[:,4] *= 2
        motion_vector[:,5] *= 2

        # Variables d'état pour la boucle
        nb_iter = 0
        coordinates = coords_col_major((rows, cols))
        imgs_registered = np.zeros([rows*cols, nb_images])
        project(coordinates, imgs_registered, images_at_level, motion_vector)
        old_imgs_a = np.zeros([rows*cols, nb_images])
        errors = np.zeros([rows*cols, nb_images])
        lagrange_mult_rho = np.zeros([rows*cols, nb_images])
        # Boucle principale
        continue_loop = True
        while continue_loop:
            # Pre-scale lambda
            lambda_value = LAMBDA/np.sqrt(rows)

            # mise à jour de A avec l'approximation de faible rang
            imgs_a = imgs_registered + errors + lagrange_mult_rho
            (U,S,V) = np.linalg.svd(imgs_a, full_matrices=False)
            for i in range(S.shape[0]):
                S[i] = shrink(S[i], 1/RHO)
            imgs_a = np.dot(U, np.dot(np.diag(S), V))

            # mise à jour de e : L1-regularized least-squares
            errors_temp = imgs_a - imgs_registered - lagrange_mult_rho
            for i in range(rows*cols):
                for j in range(nb_images):
                    errors[i,j] = shrink(errors_temp[i,j], lambda_value/RHO)

            # mise à jour de theta : forwards compositional step of a Gauss-Newton approximation.
            residuals = errors_temp - errors
            for i in range(nb_images):
                # Calcul du gradient de l'image i
                gradient = gradients.centered(imgs_registered[:,i].reshape([rows,cols]))

                # Calcul residuals et vecteur de mouvement pour l'image i
                step_params = forwards_compositional_step((rows,cols), coordinates, residuals, gradient)

                # Mise à jour de la matrice de mouvement
                motion_vector[i] = projection_params(projection_mat(motion_vector[i]) * projection_mat(step_params))

                # Transformation des paramètres de mouvement pour que la première image soit la référence
                inverse_motion_ref = np.linalg.inv(projection_mat(motion_vector[0]))
                for i in range(1,nb_images):
                    motion_vector[i] = projection_params(np.dot(inverse_motion_ref, projection_mat(motion_vector[i])))

                # Mise à jour de imgs_registered
                project(coordinates, imgs_registered, images_at_level, motion_vector)

                # y-update : dual ascent
                lagrange_mult_rho += imgs_registered - imgs_a + errors

                # Test de convergence
                residual = np.linalg.norm(imgs_a - old_imgs_a) / 1e-12.max(np.linalg.norm(old_imgs_a)) # TODO comprendre d'où vient 1e-12
                continue_loop = residual > TRESHOLD and nb_iter < MAX_ITER

                # Mise à jour des variables d'état
                nb_iter += 1
                old_imgs_a = imgs_a
        
    # Affiche le vecteur de mouvement final
    print("Vecteur de mouvement final:")
    print(motion_vector)
        


main()
