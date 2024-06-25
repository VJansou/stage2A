import cv2
import os
import numpy as np
import scipy.linalg

from consts import *
import multires
import gradients
import interpolation
import affine2d
import sparse
import utils

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

# Sauvegarde les images dans un répertoire
def store_images(images, directory):
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(directory, f"image_{i}.png"), img)

# Réduit les valeurs vers 0
def shrink(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


# Calcule le pas de Gauss-Newton
def forwards_compositional_step(shape, coordinates, residuals, gradient):
    # Initialisation des paramètres de descente
    (height, width) = shape
    descent_params = np.zeros(6)
    hessian = np.zeros((6,6))
    border = int(0.04 * min(height, width))
    pixel_count_inside = 0

    for (coord, res, grad) in zip(coordinates, residuals, gradient):
        x, y = coord
        gx, gy = grad
        # Vérifie si le pixel est dans la région d'intérêt
        if x > border and x < width - border and y > border and y < height - border:
            pixel_count_inside += 1

            # Calcul de l'approximation de Gauss-Newton
            jacobian = np.array([x*gx, x*gy, y*gx, y*gy, gx, gy])
            hessian += np.outer(jacobian, jacobian)

            # Calcul de la descente
            descent_params += res * jacobian
    if pixel_count_inside < 6:
        raise ValueError(f"Pas assez de pixels dans la région d'intérêt : {pixel_count_inside} trouvés")
    hessian_chol = scipy.linalg.cho_factor(hessian)
    return scipy.linalg.cho_solve(hessian_chol, descent_params)

# Projette les coordonnées des pixels sur l'image
def project(coordinates, registered, imgs, motion_vector):
    for (i, motion) in enumerate(motion_vector):
        motion_mat = affine2d.projection_mat(motion)
        registered_col = registered[:,i]
        for (x, y), pixel in zip(coordinates, range(len(registered_col))):
            new_pos = np.dot(motion_mat, np.array([x, y, 1.0]))
            interp = interpolation.linear(new_pos[0], new_pos[1], imgs[i])
            registered_col[pixel] = interp

# Reprojette les images en fonction du vecteur de mouvement
def reproject(imgs, motion_vector):
    reproject_imgs = []
    for i, img in enumerate(imgs):
        reproject_imgs.append(wrap(img, motion_vector[i]))
    return reproject_imgs

# Applique une transformation affine à une image
def wrap(img, motion_params):
    # Obtenir les dimensions de l'image
    if img.ndim == 2:  # Image en niveaux de gris
        rows, cols = img.shape
        channels = 1
    elif img.ndim == 3:  # Image en couleur
        rows, cols, channels = img.shape
    else:
        raise ValueError("L'image doit être 2D (niveaux de gris) ou 3D (couleur)")

    motion_mat = affine2d.projection_mat(motion_params)
    wrapped = np.zeros_like(img)
    for x in range(cols):
        for y in range(rows):
            new_pos = np.dot(motion_mat, np.array([x, y, 1.0]))
            # Appliquer l'interpolation pour chaque canal
            for c in range(channels):
                if channels == 1:
                    wrapped[y, x] = interpolation.linear(new_pos[0], new_pos[1], img)
                else:
                    wrapped[y, x, c] = interpolation.linear(new_pos[0], new_pos[1], img[:, :, c])
    return wrapped


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
    dataset_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dataset]
    if DEBUG:
        # Affiche la première image
        cv2.imshow("Premiere image en niveau de gris", dataset_gray[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Génère les images multirésolution
    image_pyramid, LEVELS = multires.mean_pyramid(MAX_LEVELS, dataset_gray)
    if DEBUG:
        # Affiche les dimensions de la première image à chaque niveau
        for level, images_at_level in enumerate(image_pyramid):
            print(f"Niveau {level}:")
            print(f"  Image 0: Dimensions {images_at_level[0].shape}")
        multires.show_first_image(image_pyramid)

    # Choisit un sous-ensemble de pixels à utiliser
    multires_sparse_pixels = [[None for _ in range(nb_images)] for _ in range(LEVELS)]
    gradients_pyramid = [[None for _ in range(nb_images)] for _ in range(LEVELS)]
    for level, images_at_level in enumerate(image_pyramid):
        gradients_pyramid[level] = [gradients.squared_norm_direct(img) for img in images_at_level]
    for i in range(nb_images):
        grad_i = []
        for level in range(LEVELS):
            grad_i.append(gradients_pyramid[level][i])
            # if DEBUG:
            #     print(f"rows : {grad_i[level].shape[0]} cols : {grad_i[level].shape[1]}")
        multires_mask = sparse.select(SPARSE_RATIO_THRESHOLD, grad_i)
        for level in range(LEVELS):
            multires_sparse_pixels[level][i] = multires_mask[level]
    
    if DEBUG:
        # Affiche le masque de sparsité pour la première image
        for level, mask_at_level in enumerate(multires_sparse_pixels):
            print(f"Niveau {level}:")
            print(f"  Image 0: {np.sum(mask_at_level[0])} pixels")
        sparse.show_first_image(image_pyramid, multires_sparse_pixels)
    
    # Initialisation du vecteur de mouvement
    motion_vector = np.zeros([nb_images,6])

    # Algorithme multi-résolution.
    # Effectue la même opération à chaque niveau pour les images et gradients correspondants.
    # L'itérateur est inversé pour commencer par le dernier niveau (la résolution la plus basse).
    # Le niveau 0 correspond aux images initiales.
    image_pyramid_inv = image_pyramid[::-1]

    # On ne garde que le résultat de la première image
    multires_sparse_pixels = [multires_sparse_pixels[level][0] for level in range(LEVELS)]
    if DEBUG:
        print("Taille de multires_sparse_pixels : ", len(multires_sparse_pixels))
    multires_sparse_pixels_inv = multires_sparse_pixels[::-1]

    for (level, (images_at_level, lvl_sparse_pixels)) in enumerate(zip(image_pyramid_inv, multires_sparse_pixels_inv)):
        if DEBUG:
            print(f"Calcul au niveau {level}")
        (rows, cols) = images_at_level[0].shape

        # Adaptation du vecteur de mouvement au changement de résolution
        motion_vector[:,4] *= 2
        motion_vector[:,5] *= 2

        # Filtre de sparsité
        pixels_count = rows * cols
        sparse_count = sum(1 if x else 0 for x in np.array(lvl_sparse_pixels).flatten())
        sparse_ratio = sparse_count / pixels_count
        if sparse_ratio > SPARSE_RATIO_THRESHOLD:
            print(f"Ratio de sparsité = {sparse_ratio} > {SPARSE_RATIO_THRESHOLD} : on utilise une résolution dense")
            sparsity = sparse.Sparsity.Full
            actual_pixel_count = pixels_count
            pixel_coordinates = utils.coords_col_major((rows, cols))
        else:
            print(f"Ratio de sparsité = {sparse_ratio} <= {SPARSE_RATIO_THRESHOLD} : on utilise une résolution sparse")
            sparsity = sparse.Sparsity.Sparse
            actual_pixel_count = sparse_count
            pixel_coordinates = utils.coordinates_from_mask(lvl_sparse_pixels)

        # Variables d'état pour la boucle
        nb_iter = 0
        coordinates = list(pixel_coordinates)
        imgs_registered = np.zeros([actual_pixel_count, nb_images])
        project(coordinates, imgs_registered, images_at_level, motion_vector)
        old_imgs_a = np.zeros([actual_pixel_count, nb_images])
        errors = np.zeros([actual_pixel_count, nb_images])
        lagrange_mult_rho = np.zeros([actual_pixel_count, nb_images])
        # Boucle principale
        continue_loop = True
        while continue_loop:
            if DEBUG:
                print(f"Iteration {nb_iter} au niveau {level}")
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
            for i in range(actual_pixel_count):
                for j in range(nb_images):
                    errors[i,j] = shrink(errors_temp[i,j], lambda_value/RHO)

            # mise à jour de theta : forwards compositional step of a Gauss-Newton approximation.
            residuals = errors_temp - errors
            for i in range(nb_images):
                # Calcul du gradient de l'image i
                match sparsity:
                    case sparse.Sparsity.Full:
                        gradient = gradients.compute_registered_gradients_full((rows,cols), imgs_registered[:,i])
                    case sparse.Sparsity.Sparse:
                        gradient = gradients.compute_registered_gradients_sparse(images_at_level[i], affine2d.projection_mat(motion_vector[i]), coordinates)

                # Calcul residuals et vecteur de mouvement pour l'image i
                step_params = forwards_compositional_step((rows,cols), coordinates, residuals[:,i], gradient.reshape(-1, 2))

                # Mise à jour de la matrice de mouvement
                motion_vector[i] = affine2d.projection_params(affine2d.projection_mat(motion_vector[i]) * affine2d.projection_mat(step_params))

                # Transformation des paramètres de mouvement pour que la première image soit la référence
                inverse_motion_ref = np.linalg.inv(affine2d.projection_mat(motion_vector[0]))
                for i in range(1,nb_images):
                    motion_vector[i] = affine2d.projection_params(np.dot(inverse_motion_ref, affine2d.projection_mat(motion_vector[i])))

                # Mise à jour de imgs_registered
                project(coordinates, imgs_registered, images_at_level, motion_vector)

                # y-update : dual ascent
                lagrange_mult_rho += imgs_registered - imgs_a + errors

                # Test de convergence
                residual = np.linalg.norm(imgs_a - old_imgs_a) / max(1e-12, np.linalg.norm(old_imgs_a))
                continue_loop = residual > TRESHOLD and nb_iter < MAX_ITER

                # Mise à jour des variables d'état
                nb_iter += 1
                old_imgs_a = imgs_a
        
    # Affiche le vecteur de mouvement final
    print("Vecteur de mouvement final:")
    print(motion_vector)
    print(np.round(motion_vector, 4))

    # Sauvegarde des images
    if SAVE_IMAGES:
        # Projection des images en fonction du vecteur de mouvement
        print("Reprojection des images...")
        registered_imgs = reproject(dataset, motion_vector)

        # Création du répertoire de sortie
        print("Sauvegarde des images...")
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        store_images(registered_imgs, OUT_DIR)
        


main()
