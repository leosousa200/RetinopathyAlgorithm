import tensorflow as tf
import numpy as np

HIDDEN = 128
DIMX = 64
DIMY = 64

def prep_translearn(model: tf.keras.Model, top_layers_to_cut: int, out_dim: int, learning_rate: float) -> tf.keras.Model:
    """
    Prepares a model for transfer learning by modifying the top layers and adding new ones.

    Args:
        model (tf.keras.Model): The original pre-trained model.
        top_layers_to_cut (int): The number of top layers to remove from the original model.
        out_dim (int): The output dimension of the new model (number of classes).
        learning_rate (float): The learning rate for model training.

    Returns:
        tf.keras.Model: The modified model ready for transfer learning.
    """
    # Define a new input layer that matches the input shape of the original model
    new_input = tf.keras.Input(shape=(DIMX, DIMY, 3), name="NEW_input")

    # Reconnect all layers of the original model to the new input, excluding the top layers
    x = new_input
    for layer in model.layers[1:-top_layers_to_cut]:
        layer.trainable = False  # Ensure the layer is not trainable
        # for BatchNormalization in finetuning if unfreezed
        x = layer(x, training=False)

    # Add new layers for fine-tuning
    x = tf.keras.layers.Flatten(name="NEW_FLAT")(x)
    x = tf.keras.layers.Dense(
        HIDDEN, activation='relu', name="NEW_Signature")(x)
    output = tf.keras.layers.Dense(
        out_dim, activation='softmax', name="NEW_output")(x)

    # Create a new model
    model = tf.keras.models.Model(inputs=new_input, outputs=output)

    compile_model(model, learning_rate)
    # show_layers(model)
    model.summary(show_trainable=True)
    return model





def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """
    Compiles the given model with a specified learning rate.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): The learning rate to use for the optimizer.

    Returns:
        None
    """
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning_rate), loss='categorical_crossentropy', 
        metrics=['accuracy'])
    return model


def unfreeze_layers(base_model: tf.keras.Model, model: tf.keras.Model, layers: int, learning_rate: float):
    # later used for fine-tuning
    # not yet implemented
    return model


def show_layers(model: tf.keras.Model):
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)