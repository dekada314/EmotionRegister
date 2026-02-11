from keras.layers import (  # noqa
    Conv2D,
    Flatten,
    Activation,
    Dropout,
    MaxPooling2D,
    Rescaling,
    RandomContrast,
    RandomTranslation,
    RandomFlip,
    RandomZoom,
    BatchNormalization,
    Dense,
)
import tensorflow as tf

from keras import Sequential
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory


TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMAGE_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 30


class EmotionRecognizer:
    def __init__(self, image_size=48, batch_size=32, epochs=30, validation_rate=0.2):
        self.size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_rate = validation_rate
        
    def get_data_augmentation(self):
        return Sequential(
        [
            Rescaling(1.0 / 255),
            RandomFlip("horizontal"),
            RandomZoom(0.15),
            RandomContrast(0.15),
            RandomTranslation(0.15, 0.15),
        ], name='data_augmentation')

    def datapretrain(self, train_dir: str, test_dir: str) -> None:
        train_ds = image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            shuffle=True,
            subset='training',
            seed=1,
        )

        validation_ds = image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            shuffle=False,
            subset='validation',
            seed=1
        )

        test_ds = image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            image_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        
        # data_aug = self.get_data_augmentation()

        # AUTOTUNE = tf.data.AUTOTUNE #noqa

        # train_ds = train_ds.map(
        #     lambda x, y: (Rescaling(1.0 / 255)(data_aug(x, training=True)), y),
        #     num_parallel_calls=AUTOTUNE
        # ).prefetch(AUTOTUNE)
    
        # validation_ds = validation_ds.map(
        #     lambda x, y: (Rescaling(1.0 / 255)(x), y),
        #     num_parallel_calls=AUTOTUNE
        # ).prefetch(AUTOTUNE)
        
        # train_ds = train_ds.map(
        #     lambda x, y: (Rescaling(1.0 / 255)(x), y),
        #     num_parallel_calls=AUTOTUNE
        # ).prefetch(AUTOTUNE)

        

        return train_ds, validation_ds, test_ds

    def create_model(self):
        model = Sequential(
            [
                Conv2D(
                    64,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=l2(0.0005),
                    input_shape=(self.size, self.size, 1),
                ),
                BatchNormalization(),
                Activation("relu"),
                Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(0.0005)),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                
                Conv2D(
                    128,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=l2(0.0005),
                    input_shape=(self.size, self.size, 1),
                ),
                BatchNormalization(),
                Activation("relu"),
                Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(0.0005)),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D(2, 2),
                Dropout(0.35),
                

                
                # ----------Classification------------

                Flatten(),
                Dense(256, kernel_regularizer=l2(0.0005)),
                BatchNormalization(),
                Activation("relu"),
                Dropout(0.5),
                Dense(7, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


if __name__ == "__main__":
    base_model = EmotionRecognizer(
        image_size=48, batch_size=32, epochs=10, validation_rate=0.2
    )
    train_data, validation_data, test_data = base_model.datapretrain(
        TRAIN_DIR, TEST_DIR
    )

    model = base_model.create_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", restore_best_weights=True, patience=12, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=0.000001, verbose=1
        ),
        ModelCheckpoint(
            monitor="val_accuracy",
            filepath="best_model.keras",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=base_model.epochs,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuracy: {test_acc:.4f}")