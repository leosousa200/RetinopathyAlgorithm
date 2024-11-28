import tensorflow as tf
import numpy as np
from tensorflow.image import resize

HIDDEN = 128
DIMX = 224
DIMY = 224

def prep_translearn(model: tf.keras.Model, top_layers_to_cut: int, out_dim: int, learning_rate: float) -> tf.keras.Model:
    """
    Prepares a model for transfer learning by modifying the top layers and adding new ones.
    """
    # Define a new input layer matching the original model's input
    new_input = tf.keras.Input((DIMX,DIMY,3), name="NEW_input")

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
    # x = tf.keras.layers.Dropout(0.3, name="NEW_Dropout")(x)  # Dropout layer added manualy
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
        
        
def augment_images(image: tf.Tensor) -> tuple[tf.Tensor]:
    """
    Applies random augmentations to an image tensor to enhance the dataset diversity.

    Args:
        image (tf.Tensor): A tensor representing an image.
        label (tf.Tensor): A tensor representing the corresponding label.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the augmented image and the original label.
    """

    # Apply random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Similarly, apply other augmentations
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Ensure pixel values remain between 0 and 1
    image = tf.clip_by_value(image, 0, 1)

    return image



def build_dataset_aug(x,y,repeat=True,batch=16, shuffle = True, aug_image_num = 0):

    '''
    Returns a tensorflow batched dataset, that can be used
    in the keras.model.fit method

        Parameters:
                x : numpy.ndarray
                        a numpy array with image filenames
                        e.g
                        array(['C:\\path\\to\\file\\dog.11481.jpg',
                               'C:\\path\\to\\file\\cat.680.jpg',
                               'C:\\path\\to\\file\\cat.9695.jpg'],
                                dtype='<U59')
                
                y : numpy.ndarray
                        one hot encoded labels
                        e.g
                        array([[0., 1.],
                               [1., 0.],
                               [1., 0.]], dtype=float32)
                repeat : boolean
                        True : when iterated over , the dataset starts generating batches
                        from the beginning if the end is reached. Suitable
                        for training set. 
                        False : will not start afresh, if iteration reaches end. suitable for
                        validation and test set
                batch : int
                        the number of samples in each batch the dataset generates when iterated.
                shuffle : boolean
                        True : Shuffle all the samples once epoch end is reached.
                        False : do not shuffle
    
        Returns:
                dataset : tensorflow.python.data.ops.dataset_ops.BatchDataset
                
    '''
    
    images = tf.constant(x) 
    labels = tf.constant(y)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(y.shape[0],reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
        
        # add augmented images
    if aug_image_num > 0:
        # Sample a subset for augmentation
        augmented = train_dataset.take(aug_image_num)
        # Apply augmentation
        augmented = augmented.map(_augment_images)
        rotation = tf.keras.layers.RandomRotation(0.15)
        augmented = augmented.map(lambda x, y: (rotation(x), y))

        train_dataset = train_dataset.concatenate(augmented)
        train_dataset = train_dataset.shuffle(
        buffer_size=train_dataset.cardinality())
    
    
    dataset = dataset.map(parse_function)
    dataset=dataset.batch(batch)

    return dataset




def parse_function(filename, label):
    
    img = tf.io.read_file(filename)
    img= tf.image.decode_image(img,channels=3,expand_animations = False)
    #img=resize(img, (64,64))
    img=resize(img, (224,224))
    img = tf.cast(img, tf.float32)/255
    return img, label