import exifread
import os
import cv2
import numpy as np
from scipy import stats
from multiprocessing import Pool

from consts import *
import exifread
import sys
import contextlib

@contextlib.contextmanager
def suppress_all_output():
    """
    A context manager to suppress stdout and stderr.
    """
    new_stdout = os.devnull
    new_stderr = os.devnull
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(new_stdout, 'w') as devnull_out:
            with open(new_stderr, 'w') as devnull_err:
                sys.stdout = devnull_out
                sys.stderr = devnull_err
                yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def load_image(filepath):
    """
    Charge une image depuis le chemin spécifié.
    Renvoie l'image et la date de prise de vue si disponible.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Impossible de charger l'image", filepath)
    
    # Lire les métadonnées de l'image
    with open(filepath, 'rb') as f:
        try:
            with suppress_all_output():
                tags = exifread.process_file(f)
            date_time_taken = tags.get('EXIF DateTimeOriginal')
        except:
            date_time_taken = None
    
    return img, date_time_taken
def load_images(directory, num_workers=4):
    """
    Charge toutes les images d'un répertoire dans une liste en utilisant le parallélisme.
    """
    filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    with Pool(num_workers) as p:
        images_metadata = p.map(load_image, filepaths)

    for _, datetime_taken in images_metadata:
        if datetime_taken:
            print(f"Date de prise de vue : {datetime_taken.values}")
        else:
            print("Date de prise de vue non disponible")
    print("")

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

def sort_images_by_date(images_metadata):
    """
    Trie les images par date de prise de vue.
    Si une des images n'a pas de date de prise de vue, affiche un message et ne trie pas.
    Renvoie seulement les images triées. (pas les dates de prise de vue)
    """
    if any([datetime_taken is None for _, datetime_taken in images_metadata]):
        print("Certaines images n'ont pas de date de prise de vue. Impossible de trier.")
        return [img for img, _ in images_metadata]   
     
    sorted_images_metadata = sorted(images_metadata, key=lambda x: x[1].values)
    return [img for img, _ in sorted_images_metadata]

if __name__ == "__main__":
    # Charger les images
    dataset = load_images(IN_DIR)#, multiprocessing.cpu_count())
    nb_images = len(dataset)
    ("Nombre d'images chargées :", nb_images)

    # Comparer les dates de prise de vue des images
    sort_images_by_date(dataset)
