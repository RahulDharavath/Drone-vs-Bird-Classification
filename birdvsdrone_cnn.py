"""---
# Transfer Learning
---

# Setup Kaggle in Colab
"""

!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

"""#  Download Dataset & Unzip:"""

!kaggle datasets download -d muhammadsaoodsarwar/drone-vs-bird -p ./ --unzip

"""---
#STEP 1: Import Libraries
---
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""---
# Check the Image Formats
---
"""

def count_images(pathdir):
    counter = Counter()   # 1
    total = 0

    for subdir, _, files in os.walk(pathdir):  # 2
        for f in files:
            ext = os.path.splitext(f)[1].lower()   # 3
            if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']:
                counter[ext] += 1
                total += 1

    print("Total images:", total)
    print(counter)

count_images("/content/dataset")

# comments
# 1. --> "Counter" → It counts how many .jpg files, .jpeg, .png files are there
# 2. --> "subdir" → the current folder it’s looking at.
#    --> "_" → a placeholder for subfolders.
#    --> "files" → a list of all files in the current folder.
# 3. -->  os.path.splitext(f) → splits the file name into:
#         i)-> "name" → cat.jpg → cat.
#         ii)-> "extension" → cat.jpg → .jpg.
#         iii) [1] → we only take the extension.
#         iv) "lower()" → converts .JPG into .jpg

"""---
# Remove webp
---
"""

def remove_webp(root_folder):
    """Remove all WebP files from dataset recursively"""
    removed = 0

    for subdir, _, files in os.walk(root_folder):
        for f in files:
            if f.lower().endswith('.webp'):
                webp_path = os.path.join(subdir, f)
                os.remove(webp_path)
                removed += 1
                print(f" Removed: {webp_path}")

    print(f"\n Total WebP files Removed: {removed}")

# Run the function
remove_webp("/content/dataset")

"""---
# Fix Corrupt Images
---
"""

import os
from PIL import Image

dataset_path = "/content/dataset"

fixed = 0
deleted = 0

for root, _, files in os.walk(dataset_path):
  for f in files:
    file_path = os.path.join(root, f)

    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
      try:
        img = Image.open(file_path)
        img.save(file_path, "JPEG", quality=95, subsampling=0)
        fixed += 1

      except Exception:
        os.remove(file_path)
        deleted += 1

print("Fixed images:", fixed)
print("Deleted unreadable images:", deleted)

"""---
# STEP 2: Load Dataset
---
"""

train_ds = keras.utils.image_dataset_from_directory(
    directory = "/content/dataset",
    validation_split = 0.2,         # 20% for validation
    subset = "training",            # Take 80% for training
    label_mode = "int",             # Returns integer labels (0,1)
    batch_size = 32,
    image_size = (224,224),
    seed = 123
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = "/content/dataset",
    validation_split = 0.2,
    subset = "validation",         # Take 20% for validation
    label_mode = "int",
    batch_size = 32,              # the model it processes images in batches.
    image_size = (224,224),
    seed = 123
)

print("Classes:", train_ds.class_names)

"""---
# STEP 3: Preprocessing (MobileNetV2)
---
"""

def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

# Apply normalization to datasets:
train_ds = train_ds.map(preprocess, num_parallel_calls= tf.data.AUTOTUNE)
validation_ds = validation_ds.map(preprocess, num_parallel_calls= tf.data.AUTOTUNE)

"""---
#STEP 4: OPTIMIZE PERFORMANCE (prefetch)
---
* for Faster training.
* When to Use:
  * Dataset is large. (10k-50k)
  * Transfer learning.
  * Video frames.
"""

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size= AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size= AUTOTUNE)

"""---
#STEP 5: Add Augmentation Using `Keras Preprocessing Layers` --> runs on GPU.
---

## `ImageDataGenerator` --> which runs on the CPU.
---
## Why is it Imp?
* Makes model robust.
* Prevents overfitting.
"""

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2)
])

def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

"""---
# STEP 6: Load Pretrained Model -> MobileNetV2 (Transfer Learning)
---
"""

conv_base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
# here we freezing conv(not changing anything in MobileNetV2 conv part)
conv_base.trainable = False

conv_base.summary()

"""---
# STEP 7: Build Model
---
## Global Average Pooling:
 * Fewer parameters
 * Less overfitting
 * Better generalization
"""

inputs = keras.Input(shape=(224, 224, 3))

x = conv_base(inputs, training= False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.summary()

"""---
# STEP 8: Compile Model
---
"""
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy",
             keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall"),
             keras.metrics.AUC(name="auc")
             ]
)

"""# STEP 9: Add EarlyStopping"""

callback = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,                # Stop after 5 epochs no improvement
    restore_best_weights=True,   # Revert to best model
    verbose=1
)

# It automatically reduces the learning rate when your model stops improving.
# Dynamically adjusts the learning rate if the model gets "stuck".
lr_scheduler = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.2,
    patience = 3,
    min_lr = 1e-6,
    verbose=1
)

"""# STEP 10: Train Model"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
history = model.fit(train_ds,
                     epochs = 35,
                     validation_data = validation_ds,
                     callbacks= [callback, lr_scheduler])



loss, acc, prec, rec, auc_score = model.evaluate(validation_ds)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall: {rec:.4f}")
print(f"Validation AUC: {auc_score:.4f}")

"""# STEP 11: Plot Results"""

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

plt.plot(history.history['precision'], label='train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Precision")
plt.legend()
plt.show()

plt.plot(history.history['recall'], label= 'Train Recall')
plt.plot(history.history['val_recall'], label= 'Validation Recall')
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.title("Training vs Validation Recall")
plt.legend()
plt.show()

"""# Test the Model"""

IMG_SIZE = (224,224)

def prepare_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

img = prepare_image("/content/dataset/bird/singleBirdinsky108_brighter.jpg")
# img = prepare_image("/content/dataset/drone/100302.jpg")
# img = prepare_image("/content/dataset/drone/000000000875.jpg")

pred = model.predict(img)

print(float(pred[0][0]))

label = "Drone" if pred > 0.9 else "Bird"
confidence = pred if pred > 0.9 else 1 - pred

print(f"Prediction: {label} | Confidence: {confidence[0][0]:.3f}")

"""# Save the Model"""
model.save("Drone_Vs_Bird_Model.keras")

