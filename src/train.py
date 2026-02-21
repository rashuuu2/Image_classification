import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os



def compile_model(model, lr=1e-5):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model



def get_class_weights(train_ds):

    print("Calculating class weights...")

    all_labels = []

    
    for images, labels in train_ds:
        # labels are one-hot encoded
        all_labels.extend(np.argmax(labels.numpy(), axis=1))

    all_labels = np.array(all_labels)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels
    )

    class_weights = dict(enumerate(class_weights))

    print("Class Weights:", class_weights)
    return class_weights



def train_model(model, train_ds, val_ds, epochs):

    class_weights = get_class_weights(train_ds)

    os.makedirs("artifacts", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "artifacts/best_model.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    return history



if __name__ == "__main__":

    from model import build_model
    from data_loader import load_data

    DATA_PATH = "data/raw"   # dataset folder
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 2
    EPOCHS = 10

    print("\nLoading datasets...")
    train_ds, val_ds = load_data(DATA_PATH, IMG_SIZE, BATCH_SIZE)

    print("\nBuilding model...")
    model = build_model(NUM_CLASSES, IMG_SIZE)

    print("\nCompiling model...")
    model = compile_model(model)

    print("\nStarting training...\n")
    history = train_model(model, train_ds, val_ds, EPOCHS)

    print("\n✅ Training Finished!")
