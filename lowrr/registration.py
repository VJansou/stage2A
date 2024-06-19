# Algortihme de registration pour une séquence d'images légèrement décalées.

def run(dataset):
    # Récupérer le nombre d'images
    nb_images = len(dataset)

    # Initialiser les images multirésolution et la norme des gradients
    