import tensorflow as tf
import numpy as np

HIDDEN = 128
DIMX = 64
DIMY = 64

def prep_translearn(model: tf.keras.Model, top_layers_to_cut: int, out_dim: int, learning_rate: float) -> tf.keras.Model:
    """
    Prepares a model for transfer learning by modifying the top layers and adding new ones.
    """
    # Define a new input layer matching the original model's input
    new_input = tf.keras.Input(shape=model.input_shape[1:], name="NEW_input")

    # Reconnect layers from the original model
    x = new_input
    for i, layer in enumerate(model.layers[:-top_layers_to_cut]):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            x = layer(x, training=False)
        else:
            x = layer(x)
        print(f"After layer {i} ({layer.name}), shape: {x.shape}")

    # Add new fine-tuning layers
    x = tf.keras.layers.Flatten(name="NEW_FLAT")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="NEW_Signature")(x)
    output = tf.keras.layers.Dense(out_dim, activation='softmax', name="NEW_output")(x)

    # Create and compile the new model
    new_model = tf.keras.models.Model(inputs=new_input, outputs=output)
    compile_model(new_model, learning_rate)

    return new_model






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