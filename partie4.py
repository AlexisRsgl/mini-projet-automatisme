
# PARTIE 2 + PARTIE 3 + PARTIE 4 : CNN Cats vs Dogs

# IMPORT DES BIBLIOTHÈQUES
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
import matplotlib.pyplot as plt

# PARAMÈTRES GÉNÉRAUX
DATASET_DIR = "Asirra_ cat vs dogs"  # chemin vers le dossier contenant cats/ et dogs/
IMG_SIZE = (150, 150)                # taille cible des images
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")  # extensions acceptées

# CHARGEMENT DES IMAGES ET DES LABELS
image_paths = []
labels = []

for img_name in os.listdir(DATASET_DIR):
    if not img_name.lower().endswith(VALID_EXTENSIONS):
        continue
    if img_name.lower().startswith("cat"):
        label = 0
    elif img_name.lower().startswith("dog"):
        label = 1
    else:
        continue
    image_paths.append(os.path.join(DATASET_DIR, img_name))
    labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Total images :", len(image_paths))

# SÉPARATION TRAIN / VALIDATION / TEST
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

print("Train :", len(X_train))
print("Validation :", len(X_val))
print("Test :", len(X_test))

# FONCTION DE CHARGEMENT ET PRÉTRAITEMENT DES IMAGES
def load_and_preprocess_images(paths):
    images = []
    for path in paths:
        try:
            img = load_img(path, target_size=IMG_SIZE)
            img = img_to_array(img)
            images.append(img)
        except:
            continue
    return np.array(images)

X_train_img = load_and_preprocess_images(X_train)
X_val_img = load_and_preprocess_images(X_val)
X_test_img = load_and_preprocess_images(X_test)

# IMAGE DATA GENERATOR
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train_img, y_train, batch_size=32, shuffle=True)
validation_generator = val_datagen.flow(X_val_img, y_val, batch_size=32, shuffle=False)
test_generator = test_datagen.flow(X_test_img, y_test, batch_size=32, shuffle=False)

# Optionnel : afficher 9 images pour vérifier
def plot_images(generator):
    images, labels = next(generator)
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow((images[i] * 255).astype('uint8'))
        plt.title('Chat' if labels[i]==0 else 'Chien')
        plt.axis('off')
    plt.show()

plot_images(train_generator)

# CONSTRUCTION DU CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ENTRAÎNEMENT DU MODELE
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=10
)

# VISUALISATION DES COURBES
# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
