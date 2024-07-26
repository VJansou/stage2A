# Constantes utilisées dans le programme
# IN_DIR = "Dataset/id_pose_8_19_25_33" # Dossier contenant les images d'entrée (bear)
# IN_DIR = "Dataset/id_pose_5_10_15_20" # Dossier contenant les images d'entrée (bouddha)
# IN_DIR = "Dataset/id_pose_10_20_30_40" # Dossier contenant les images d'entrée (pot)
# IN_DIR = "Dataset/id_pose_5_9_15" # Dossier contenant les images d'entrée (reading)
# IN_DIR = "Dataset/ASupp" # Les images d'une même pose sont les mêmes
IN_DIR = "Dataset/id_pose_4_8_12_16_20_24" # Dossier contenant les images d'entrée (guerrier)
# IN_DIR = "Dataset/id_pose_9_18_27_36_45" # Dossier contenant les images d'entrée (Auguste)

CROP_FACTOR = 0.6 # Facteur de recadrage des images (1 = pas de recadrage)

MAX_LEVELS = 7 # Nombre de niveaux max dans la pyramide
LEVEL_USED = 4 # Niveau de la pyramide utilisé pour les calculs
NB_PIXELS_MIN = 50 # Nombre de pixels par ligne/colonne dans l'image après pyramide

SEUIL_PIXEL_DIFF = 8 # Seuil de différence de pixel pour considérer deux pixels comme différents
SEUIL = 0.7 # Seuil de changement de pose entre les images

DEBUG = False # Mode débogage