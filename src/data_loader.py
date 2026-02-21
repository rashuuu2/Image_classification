import tensorflow as tf
import os
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_data(data_path, img_size, batch_size):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_path, "train"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_path, "val"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical"
    )

    
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    # performance optimization (optional but good)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
