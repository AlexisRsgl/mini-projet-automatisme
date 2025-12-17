# 1. IMPORT DES BIBLIOTHÈQUES

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import Counter


# 2. CHARGEMENT DU DATASET

# Chemin vers le dossier contenant toutes les images (chats et chiens mélangés)
DATASET_DIR = "Asirra_ cat vs dogs"

# Liste qui contiendra les chemins complets des images valides
image_paths = []

# Liste qui contiendra les labels associés aux images
# 0 = chat, 1 = chien
labels = []

# Extensions d’images acceptées
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Parcours de tous les fichiers du dossier
for img_name in os.listdir(DATASET_DIR):

    # On ignore les fichiers qui ne sont pas des images
    if not img_name.lower().endswith(VALID_EXTENSIONS):
        continue

    # Construction du chemin complet vers l’image
    img_path = os.path.join(DATASET_DIR, img_name)

    # Attribution du label en fonction du nom du fichier
    # Les fichiers commencent par "cat" ou "dog"
    if img_name.lower().startswith("cat"):
        label = 0  # chat
    elif img_name.lower().startswith("dog"):
        label = 1  # chien
    else:
        continue  # on ignore les fichiers non reconnus

    # Ajout du chemin de l’image et de son label
    image_paths.append(img_path)
    labels.append(label)

# Affichage du nombre d’images valides chargées
print(f"Images valides trouvées : {len(image_paths)}")
print(f"Nombre total d'images : {len(image_paths)}")


# 3. AFFICHER 10 IMAGES ALÉATOIRES

# Création d’une figure pour afficher les images
plt.figure(figsize=(12, 6))

# Compteur du nombre d’images affichées
shown = 0

# Sélection aléatoire d’indices parmi toutes les images
for idx in random.sample(range(len(image_paths)), len(image_paths)):
    try:
        # Chargement de l’image
        img = load_img(image_paths[idx])

        # Placement de l’image dans une grille 2 lignes × 5 colonnes
        plt.subplot(2, 5, shown + 1)
        plt.imshow(img)

        # Affichage de la classe associée
        plt.title("Chat" if labels[idx] == 0 else "Chien")
        plt.axis("off")

        shown += 1

        # Arrêt après 10 images affichées
        if shown == 10:
            break

    except:
        # Ignore les images corrompues ou illisibles
        continue

# Affichage final
plt.show()


# 4. VÉRIFICATIONS

# a) Vérification des tailles des images

# Liste pour stocker les tailles (largeur, hauteur)
sizes = []

# Sélection aléatoire de 20 images maximum
for path in random.sample(image_paths, min(20, len(image_paths))):
    try:
        img = load_img(path)
        sizes.append(img.size)  # (width, height)
    except:
        continue

# Affichage des tailles observées
print("Exemples de tailles d'images :", sizes)


# b) Distribution des classes (chats vs chiens)

# Comptage du nombre d’images par classe
class_distribution = Counter(labels)

print("Distribution des classes :")
print("Chats :", class_distribution[0])
print("Chiens :", class_distribution[1])


# 5. PIPELINE DE PRÉPARATION

# Taille cible pour toutes les images
IMG_SIZE = (150, 150)

def preprocess_image(img_path):
    """
    Fonction qui prépare une image pour l'entraînement :
    - redimensionnement à 150x150
    - conversion en tableau numpy
    - normalisation des pixels entre 0 et 1
    """
    # Chargement et redimensionnement de l’image
    img = load_img(img_path, target_size=IMG_SIZE)

    # Conversion de l’image en tableau numpy
    img_array = img_to_array(img)

    # Normalisation des valeurs des pixels
    img_array = img_array / 255.0

    return img_array


# 6. TEST DU PIPELINE

# Prétraitement d’une image pour vérification
test_img = preprocess_image(image_paths[0])

# Affichage des caractéristiques de l’image traitée
print("Shape après preprocessing :", test_img.shape)
print("Min / Max :", test_img.min(), test_img.max())
