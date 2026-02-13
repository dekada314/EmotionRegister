from keras.layers import (  # noqa
    Conv2D,
    GlobalAveragePooling2D,
    Activation,
    Dropout,
    MaxPooling2D,
    Rescaling,
    RandomContrast,
    RandomFlip,
    RandomZoom,
    BatchNormalization,
    Dense,
    RandomRotation,
    RandomBrightness,
)

import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from keras import Sequential
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy
from keras.activations import softmax, relu
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import AdamW
from keras.utils import image_dataset_from_directory


TRAIN_DIR = "dataset/train_new"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

IMAGE_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15


class EmotionRecognizer:
    def __init__(self, image_size=48, batch_size=32, epochs=30):
        self.size = image_size
        self.batch_size = batch_size
        self.epochs = epochs

    def get_data_augmentation(self):
        return Sequential(
            [
                Rescaling(1.0 / 255),
                RandomFlip("horizontal"),
                RandomRotation(0.15),
                RandomZoom(0.15),
                RandomContrast(0.2),
                RandomBrightness(0.2),
            ],
            name="data_augmentation",
        )

    def datapretrain(self, train_dir: str, validation_dir: str, test_dir: str) -> None:
        train_ds = image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=self.batch_size,
            image_size=(self.size, self.size),
            shuffle=True,
        )

        validation_ds = image_dataset_from_directory(
            validation_dir,
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=self.batch_size,
            image_size=(self.size, self.size),
            shuffle=False,
        )

        test_ds = image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            color_mode="grayscale",
            batch_size=self.batch_size,
            image_size=(self.size, self.size),
        )

        data_aug = self.get_data_augmentation()

        AUTOTUNE = tf.data.AUTOTUNE  # noqa

        train_ds = (
            train_ds.map(
                lambda x, y: (data_aug(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )
            .map(lambda x, y: (x / 255, y))
            .prefetch(AUTOTUNE)
        )

        validation_ds = validation_ds.map(
            lambda x, y: (x / 255, y), num_parallel_calls=AUTOTUNE
        ).prefetch(AUTOTUNE)

        test_ds = test_ds.map(
            lambda x, y: (x / 255, y), num_parallel_calls=AUTOTUNE
        ).prefetch(AUTOTUNE)

        # def classes_count(ds, name):
        #     counts = {i: 0 for i in range(7)}
        #     for _, labels in ds.unbatch().as_numpy_iterator():
        #         cls = np.argmax(labels)
        #         counts[cls] += 1
        #     print(f"{name} class distribution:", counts)
        #     total_sum = sum(counts.values())
        #     print(
        #         f"{name} percentages: ",
        #         {k: f"{v / total_sum * 100:.1f}%" for k, v in counts.items()},
        #     )

        # print("Распределения классов:")
        # classes_count(train_ds, "training")
        # classes_count(validation_ds, "validation")
        # classes_count(test_ds, "testing")

        return train_ds, validation_ds, test_ds

    def create_model(self):
        model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=l2(0.0001),
                    input_shape=(self.size, self.size, 1),
                ),
                BatchNormalization(),
                Activation(relu),
                MaxPooling2D(),
                Dropout(0.2),
                Conv2D(
                    64,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                BatchNormalization(),
                Activation(relu),
                MaxPooling2D(),
                Dropout(0.25),
                Conv2D(
                    128,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=l2(0.001),
                ),
                BatchNormalization(),
                Activation(relu),
                MaxPooling2D(),
                Dropout(0.3),
                # ----------Classification------------
                GlobalAveragePooling2D(),
                
                Dense(128, kernel_regularizer=l2(0.001)),
                Dropout(0.4),
                
                Dense(7, activation=softmax),
            ]
        )

        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss=CategoricalCrossentropy,
            metrics=["accuracy"],
        )
        return model


if __name__ == "__main__":
    base_model = EmotionRecognizer(
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS
    )
    train_data, validation_data, test_data = base_model.datapretrain(
        TRAIN_DIR, VALIDATION_DIR, TEST_DIR
    )

    model = base_model.create_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", restore_best_weights=True, patience=12, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        ),
        # ModelCheckpoint(
        #     monitor="val_accuracy",
        #     filepath="best_model.keras",
        #     save_best_only=True,
        #     save_weights_only=False,
        #     verbose=1,
        # ),
    ]

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=base_model.epochs,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuracy: {test_acc:.4f}")
