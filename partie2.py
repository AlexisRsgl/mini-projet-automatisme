# PARTIE 2 – PRÉPARATION DES DONNÉES AVEC ImageDataGenerator

# IMPORT DES BIBLIOTHÈQUES
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 1. PARAMÈTRES GÉNÉRAUX

# Chemin vers le dossier contenant toutes les images (chats et chiens mélangés)
DATASET_DIR = "Asirra_ cat vs dogs"

# Taille cible des images (hauteur, largeur)
IMG_SIZE = (150, 150)

# Extensions d’images acceptées
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


# 2. CHARGEMENT DES IMAGES ET DES LABELS

# Liste des chemins vers les images
image_paths = []

# Liste des labels associés aux images
# 0 = chat, 1 = chien
labels = []

# Parcours de tous les fichiers du dossier
for img_name in os.listdir(DATASET_DIR):

    # Ignorer les fichiers qui ne sont pas des images
    if not img_name.lower().endswith(VALID_EXTENSIONS):
        continue

    # Attribution du label en fonction du nom du fichier
    if img_name.lower().startswith("cat"):
        label = 0
    elif img_name.lower().startswith("dog"):
        label = 1
    else:
        continue  # ignorer les fichiers non reconnus

    # Stockage du chemin complet et du label correspondant
    image_paths.append(os.path.join(DATASET_DIR, img_name))
    labels.append(label)

# Conversion en tableaux numpy
image_paths = np.array(image_paths)
labels = np.array(labels)

# Affichage du nombre total d’images chargées
print("Total images :", len(image_paths))


# 3. SÉPARATION TRAIN / VALIDATION / TEST

# Première séparation :
# 70 % pour l’entraînement
# 30 % temporaire (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels  # garantit la même proportion chats/chiens
)

# Deuxième séparation du jeu temporaire :
# 20 % validation
# 10 % test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=1/3,  # 1/3 de 30 % = 10 %
    random_state=42,
    stratify=y_temp
)

# Affichage des tailles des ensembles
print("Train :", len(X_train))
print("Validation :", len(X_val))
print("Test :", len(X_test))


# 4. FONCTION DE CHARGEMENT ET DE PRÉTRAITEMENT DES IMAGES

def load_and_preprocess_images(paths):
    """
    Cette fonction :
    - charge les images depuis leurs chemins
    - redimensionne chaque image en 150x150
    - convertit les images en tableaux numpy
    """
    images = []

    for path in paths:
        try:
            # Chargement et redimensionnement de l’image
            img = load_img(path, target_size=IMG_SIZE)

            # Conversion en tableau numpy
            img = img_to_array(img)

            images.append(img)

        except:
            # Ignore les images corrompues ou illisibles
            continue

    # Conversion finale en tableau numpy
    return np.array(images)


# Chargement des images pour chaque ensemble
X_train_img = load_and_preprocess_images(X_train)
X_val_img = load_and_preprocess_images(X_val)
X_test_img = load_and_preprocess_images(X_test)


# 5. ImageDataGenerator

# Générateur pour les données d'entraînement
# rescale=1./255 permet de normaliser les pixels entre 0 et 1
train_datagen = ImageDataGenerator(rescale=1./255)

# Générateur pour la validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Générateur pour le test
test_datagen = ImageDataGenerator(rescale=1./255)

# Création des générateurs Keras à partir des tableaux numpy
train_generator = train_datagen.flow(
    X_train_img,
    y_train,
    batch_size=32,
    shuffle=True  # mélange les données à chaque époque
)

validation_generator = val_datagen.flow(
    X_val_img,
    y_val,
    batch_size=32,
    shuffle=False
)

test_generator = test_datagen.flow(
    X_test_img,
    y_test,
    batch_size=32,
    shuffle=False
)

# Affiche 9 images du train_generator
def plot_images(generator):
    images, labels = next(generator)  # récupère un batch
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow((images[i] * 255).astype('uint8'))  # convertit pour affichage
        plt.title('Chat' if labels[i]==0 else 'Chien')
        plt.axis('off')
    plt.show()

plot_images(train_generator)
