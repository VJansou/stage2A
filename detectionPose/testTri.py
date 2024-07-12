import exifread
import os
import cv2
import numpy as np
from scipy import stats
from multiprocessing import Pool

from consts import *

def load_image(filepath):
    """
    Charge une image depuis le chemin spécifié.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Impossible de charger l'image", filepath)
    
    # Lire les métadonnées de l'image
    with open(filepath, 'rb') as f:
        tags = exifread.process_file(f)
        date_time_taken = tags.get('EXIF DateTimeOriginal')
    
    return img, date_time_taken
    

def load_images(directory, num_workers=4):
    """
    Charge toutes les images d'un répertoire dans une liste en utilisant le parallélisme.
    """
    filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    with Pool(num_workers) as p:
        images_metadata = p.map(load_image, filepaths)

    return images_metadata

def compare_photo_times(image_metadata_list):
    """
    Compare les dates de prise de vue des images et les trie par ordre chronologique.
    """
    # Trier les images par date de prise de vue
    sorted_images_metadata = sorted(image_metadata_list, key=lambda x: x[1].values if x[1] else '')

    for _, datetime_taken in sorted_images_metadata:
        if datetime_taken:
            print(f"Date de prise de vue : {datetime_taken.values}")
        else:
            print("Date de prise de vue non disponible")

if __name__ == "__main__":
    # Charger les images
    dataset = load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)
    ("Nombre d'images chargées :", nb_images)

    # Comparer les dates de prise de vue des images
    compare_photo_times(dataset)
