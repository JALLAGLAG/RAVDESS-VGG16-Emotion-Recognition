"""VGG16 model used for frame-level facial emotion recognition."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import IMG_SIZE, CHANNELS, NUM_CLASSES, LEARNING_RATE, DROPOUT_1, DROPOUT_2


def build_vgg16_model(num_classes: int = NUM_CLASSES) -> Model:
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    )

    # Fine-tune only the top convolutional block for a lightweight setup.
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(DROPOUT_1)(x)
    x = Dense(256, activation="relu", name="feature_vector")(x)
    x = Dropout(DROPOUT_2)(x)
    outputs = Dense(num_classes, activation="softmax", name="emotion_softmax")(x)

    model = Model(inputs, outputs, name="VGG16_RAVDESS_Frame_Classifier")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
