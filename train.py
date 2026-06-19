import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


def build_model(dropout_rate, learning_rate):

    model = models.Sequential([

        layers.Input(shape=(28,28,1)),

        layers.Conv2D(
            32,
            (3,3),
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2,2)),

        layers.Conv2D(
            64,
            (3,3),
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2,2)),

        layers.Conv2D(
            128,
            (3,3),
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.Flatten(),

        layers.Dense(
            128,
            activation='relu'
        ),

        layers.Dropout(dropout_rate),

        layers.Dense(
            10,
            activation='softmax'
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


configs = [

    {"dropout":0.3,"lr":1e-3},
    {"dropout":0.5,"lr":1e-3},
]

best_val_acc = 0
best_model = None
best_config = None

for config in configs:

    print("\n" + "="*50)
    print(config)
    print("="*50)

    model = build_model(
        config["dropout"],
        config["lr"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )


    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    current_val_acc = max(
        history.history["val_accuracy"]
    )

    print(
        f"Validation Accuracy: {current_val_acc:.4f}"
    )

    if current_val_acc > best_val_acc:

        best_val_acc = current_val_acc
        best_model = model
        best_config = config

print("\nBest Config:")
print(best_config)

loss, accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {accuracy:.4f}")

os.makedirs("model", exist_ok=True)

best_model.save(
    "model/mnist_cnn.keras"
)

print("\nModel Saved Successfully")