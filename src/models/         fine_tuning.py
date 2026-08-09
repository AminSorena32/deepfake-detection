import tensorflow as tf

def prepare_for_fine_tuning(
    model,
    unfreeze_layers=15,
):
    """
    Prepare the trained model for fine-tuning by
    unfreezing the last N layers of EfficientNetB0.
    """

    # Get the backbone by name instead of layer index
    base_model = model.get_layer("efficientnet_backbone")

    # Make backbone trainable
    base_model.trainable = True

    # Freeze all layers except the last N layers
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    return model


def compile_for_fine_tuning(
    model,
    learning_rate=1e-5,
):
    """
    Compile the model with a lower learning rate
    for fine-tuning.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model