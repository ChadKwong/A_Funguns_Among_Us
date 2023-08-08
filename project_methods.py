from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
tf.debugging.set_log_device_placement(True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import dask as dk
from dask_ml.model_selection import train_test_split

from memory_profiler import profile




#data loading method
@profile
def load_data(file_path, returnImg = False):
    
    #Define data output as dictionary
    data = dict()
    img = list()
    #Loop through folders
    loop = os.listdir(file_path)
    print('Importing data...')
    with tqdm(loop) as file_iterator:
        for folder in file_iterator:
            if folder != ".DS_Store":
                #Create list entry for folder in return dictionary
                if folder not in data:
                    data[folder] = list()
                #Loop through files within each folder
                new_path = os.path.join(file_path,folder)
                for file in tqdm(os.listdir(new_path)):
                    #Check for png file
                    if file.endswith(".png"):
                        #store image path for debugging
                        image_path = os.path.join(new_path, file)
                        try:
                            #Read and append image 
                            image = Image.open(image_path)
                            data[folder].append(np.array(image).tolist())
                            img.append(image)
                        #print error for any problematic images
                        except IOError:
                            print(f"Error reading image: {image_path}")
                            
    #generate return dataset including X and Y
    X = list()
    Y = list()
    print("Formatting and returning dataset...")
    for label, imgs in tqdm(data.items()):
        #map X
        X.extend(imgs)
        #map Y
        Y.extend([label] * len(imgs))
    
    print('Import complete.')
    if returnImg ==True:
        return img, X, Y
    else:
        return np.array(X), np.array(Y)

#Function for loading data into dask dataframe
@profile
def load_data_dask(file_path, partition_size=100, chunk_size=10): 
    #define internal image loading function
    def load_image(file_path):
        try:
            image = Image.open(file_path)
            return np.array(image)
        except IOError:
            print(f"Error reading image: {file_path}")
            return None
    
    #Initiate daskbag pipeline
    data = dk.bag.from_sequence(os.listdir(file_path), partition_size=partition_size)
    # Filter out .DS_Store from folders
    data = data.filter(lambda folder: folder != ".DS_Store")
    folders = data.map(lambda folder: os.path.join(file_path, folder))
    #export items from folders and flatten into one list
    items = folders.map(lambda folder: [os.path.join(folder,item) for item in os.listdir(folder)])
    #filter out non png files
    data = items.flatten().filter(lambda item: item.endswith(".png"))
    #process data into the shape (512,512,4)
    imgData = data.map(load_image).filter(lambda img: img is not None)
    
    #Export pipeline to dask array
    X = dk.array.from_array(list(imgData), chunks=chunk_size)  # This creates a Dask array

    # # Generate return dataset including X and Y
    Y = np.array([label.split('/')[1] for label in data])

    print('Import complete.\n')
    return X.compute(), np.array(Y)

#Preprocessing method
@profile
def preprocess(X, Y, split = .25, num_class = 1, rand_state = None):
    # Split the data into train and test sets (80% for training, 20% for testing)
    # Use simple random split due to balance within dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, random_state=rand_state)

    # Verify the shapes of the resulting sets
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    #Preprocessing data
    # Normalize pixel values to a range of 0 to 1
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Convert encoded labels to one-hot encoded vectors
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=num_class)
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=num_class)
    return X_train, X_test, Y_train, Y_test

#data processing with dask
@profile
def preprocess_dask(X, Y, split = .25, num_class = 1, rand_state = None, batch_size = 512):
    # Use simple random split due to balance within dataset
    # Split the data into train and test sets using Dask
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, random_state=rand_state, shuffle = True)

    # Verify the shapes of the resulting sets
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    #Preprocessing data
    # Normalize pixel values to a range of 0 to 1
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Convert encoded labels to one-hot encoded vectors
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=num_class)
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=num_class)
    
    # Prepare data for tensorflow use
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
    
    return train_dataset, test_dataset

#Model running method
@profile
def run_CNN_model(X_train, X_test, Y_train, Y_test, epoch_num=10, metric_functions = ['accuracy'],
              loss_function='categorical_crossentropy', optimizer_function = 'adam', encoder = None):
    classes = len(encoder.classes_)
    shape = tuple([s for s in X_train.shape[1:]])
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'), # Increased filters for larger images
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'), # Increased neurons in the dense layer
        layers.Dense(classes, activation='softmax')
    ])


    model.compile(optimizer=optimizer_function,
                  loss=loss_function,
                  metrics=metric_functions)

    history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=512, validation_split=0.1)
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f"Test accuracy: {test_accuracy}")
    return model

#Function with improvements for performance
@profile
def run_improved_CNN_model(train, test, epoch_num=10, metric_functions=['accuracy'],
                  loss_function='categorical_crossentropy', optimizer_function='adam', encoder=None, batch=512,
                  validation_split=0.1):
    #Calculate the number of classes
    classes = len(encoder.classes_)
    
    #Configure shape/validation size for both dask and non-dask data
    if isinstance(train, tf.data.Dataset):
        shape = train.element_spec[0].shape[1:]
        validation_size = int(validation_split * len(train))
        validation_dataset = train.take(validation_size)
        train_dataset = train.skip(validation_size)
    else:
        X_train, Y_train = train
        X_test, Y_test = test
        shape = tuple([s for s in X_train.shape[1:]])
    
    #Configure the model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),  # Increased filters for larger images
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Increased neurons in the dense layer
        layers.Dense(classes, activation='softmax')
    ])
    
    #Compile the model
    model.compile(optimizer=optimizer_function,
                  loss=loss_function,
                  metrics=metric_functions)
    
    #fit the model depending on whether the data is dask or not
    if isinstance(train, tf.data.Dataset):
        history = model.fit(train_dataset, epochs=epoch_num, batch_size=batch, validation_data=validation_dataset)
        test_loss, test_accuracy = model.evaluate(test)
        print(f"Test accuracy: {test_accuracy}")
    else:
        history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=batch, validation_split=validation_split)
        test_loss, test_accuracy = model.evaluate(X_test, Y_test)
        print(f"Test accuracy: {test_accuracy}")

    return model

#Model running method
@profile
def run_DenseNet_model(X_train, X_test, Y_train, Y_test, epoch_num=10, metric_functions = ['accuracy'],
              loss_function='categorical_crossentropy', optimizer_function = 'adam', encoder = None):
    classes = len(encoder.classes_)
    shape = tuple([s for s in X_train.shape[1:]])
        # Function to create a dense block
    def dense_block(x, num_layers, growth_rate):
        for _ in range(num_layers):
            conv = layers.Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(x)
            x = layers.concatenate([x, conv], axis=-1)
        return x

    # Function to create a transition block with 1x1 convolution and average pooling
    def transition_block(x, num_filters):
        x = layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(x)
        x = layers.AveragePooling2D((2, 2))(x)
        return x

    # Build the DenseNet model
    def build_densenet_model(input_shape = shape, num_blocks=3, num_layers_per_block=4, growth_rate=32, num_classes=classes):
        inputs = layers.Input(shape=input_shape)
        x = inputs

        # Initial convolutional layer
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Dense blocks and transition blocks
        for i in range(num_blocks):
            x = dense_block(x, num_layers_per_block, growth_rate)
            if i < num_blocks - 1:
                x = transition_block(x, x.shape[-1] // 2)  # Reduce the number of filters by half in transition blocks

        # Global average pooling and final dense layer
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, x)

        return model

    model = build_densenet_model()
    model.compile(optimizer=optimizer_function,
                  loss=loss_function,
                  metrics=metric_functions)

    history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=512, validation_split=0.1)
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f"Test accuracy: {test_accuracy}")
    return model

#Simple function for calculating and formatting predictions from models
@profile
def predictions(input_data, model, encoder = None):
    predictions = encoder.inverse_transform([np.argmax(prediction) for prediction in model.predict(input_data)])
    return predictions