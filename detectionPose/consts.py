# Constantes utilisées dans le programme
IN_DIR = "Dataset/id_pose_8_19_25_33" # Dossier contenant les images d'entrée (bear)
# IN_DIR = "Dataset/id_pose_5_10_15_20" # Dossier contenant les images d'entrée (bouddha)
# IN_DIR = "Dataset/id_pose_10_20_30_40" # Dossier contenant les images d'entrée (pot)
# IN_DIR = "Dataset/ASupp" # Les images d'une même pose sont les mêmes
# IN_DIR = "Dataset/id_pose_4_8_12_16_20_24" # Dossier contenant les images d'entrée (guerrier)

CROP_FACTOR = 0.8 # Facteur de recadrage des images (1 = pas de recadrage)

MAX_LEVELS = 7 # Nombre de niveaux max dans la pyramide
LEVEL_USED = 4 # Niveau de la pyramide utilisé pour les calculs

SEUIL = 0.7 # Seuil de changement de pose entre les images

DEBUG = False # Mode débogage