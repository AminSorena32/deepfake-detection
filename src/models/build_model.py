from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

def build_model(
    input_shape=(224, 224, 3),
    learning_rate=1e-4,
):
    """
    Build the DeepFake detection model using
    an ImageNet-pretrained EfficientNetB0 backbone.
    """

    # Pretrained EfficientNetB0 backbone
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        name="efficientnet_backbone",
    )

    # Freeze backbone for transfer learning
    base_model.trainable = False

    # Model input
    inputs = layers.Input(shape=input_shape)

    # Feature extraction
    x = base_model(inputs, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    # Final model
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="deepfake_detector",
    )

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
