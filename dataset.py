import tensorflow as tf
from tensorflow import keras

def get_datasets(data_dir, batch_size=128, img_size=(64, 64), val_split=0.15, seed=42):
    # Explicitly define classes to ignore the 'checkpoints' folder
    VALID_CLASSES = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']

    # 1. Base Datasets (We load labels here as 'int' to satisfy Keras, but will discard them later)
    train_ds_base = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="training",
        seed=seed, color_mode='grayscale', image_size=img_size, 
        batch_size=batch_size, label_mode='int', class_names=VALID_CLASSES
    )
    
    val_ds_base = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="validation",
        seed=seed, color_mode='grayscale', image_size=img_size, 
        batch_size=batch_size, label_mode='int', class_names=VALID_CLASSES
    )

    # 2. Evaluation Dataset
    val_ds_labels = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="validation",
        seed=seed, color_mode='grayscale', image_size=img_size, 
        batch_size=batch_size, label_mode='int', class_names=VALID_CLASSES
    )
    class_names = val_ds_labels.class_names

    # 3. Normalization & Mapping
    normalization_layer = keras.layers.Rescaling(1./255)

    # We now accept the (image, label) pair, but discard the label and return (image, image)
    def prepare_for_training(x, y):
        x = normalization_layer(x)
        return x, x  

    def prepare_for_eval(x, y):
        return normalization_layer(x), y

    # Apply mapping, CACHE IN RAM, and optimize performance
    train_ds = train_ds_base.map(prepare_for_training, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds_base.map(prepare_for_training, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    val_ds_eval = val_ds_labels.map(prepare_for_eval, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_eval = val_ds_eval.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, val_ds_eval, class_names
