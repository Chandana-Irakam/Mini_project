import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Path to your dataset
data_dir = r"C:\Users\sures\OneDrive\Desktop\Mini_project2\stampede"  # contains 'stampede' and 'normal' folders

# Data augmentation + normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,  # ✅ 75% train, 25% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Create training and validation generators
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Compute class weights (to fix class imbalance)
class_labels = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(class_labels), y=class_labels)
class_weights = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weights)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("stampede_model_best.h5", monitor="val_accuracy", save_best_only=True)
earlystop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=35,   # you can increase to 25 if validation improves
    class_weight=class_weights,
    callbacks=[checkpoint, earlystop]
)

print("✅ Stampede Detection Model Trained and Saved as 'stampede_model_best.h5'")
