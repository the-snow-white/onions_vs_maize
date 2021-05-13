import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import shutil
import pathlib

from keras_preprocessing.image import ImageDataGenerator, image_data_generator, load_img, img_to_array

#gpus = tf.config.experimental.list_physical_devices('CPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

#ImageDataGenerators
rotation_range = 40
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True
fill_mode ='nearest'
target_size = (150, 150)
batch_size = 20
class_mode = 'binary'

#Model Parameters
train_validation_split = 0.8
metrics = ['accuracy']
steps_per_epoch = 1000
epochs = 50
validation_steps = 500
input_shape=(150, 150, 3)


def create_folder_structure():
    print('Creating folder structure')
    shutil.rmtree('./train')
    train_paths = ["./train/train/maize","./train/train/onions","./train/validate/maize", "./train/validate/onions"]
    try:
        for path in train_paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            print('Created path  {}\n'.format(path))

    except OSError as err:  
        print(err)      


def copy_images(source_list, destination_path):
    print('Copying Images from {} to {}'.format(source_list, destination_path))
    for image in source_list:
        shutil.copyfile(f'./input_data/{image}', f'./train/{destination_path}/{image}')
        print('Copied Image')
        print(image)


def create_training_and_validation_set():
    print('Creating training and validation data')
    
    onions_and_maize_images = os.listdir('./input_data')    
    maize_images = list(filter(lambda image: 'Maize' in image, onions_and_maize_images))
    onions_images = list(filter(lambda image: 'Onion' in image, onions_and_maize_images))

    train_gen = ImageDataGenerator(
        rotation_range = rotation_range,
        width_shift_range = width_shift_range,
        height_shift_range = height_shift_range,
        shear_range = shear_range,
        zoom_range = zoom_range,
        horizontal_flip = horizontal_flip,
        fill_mode = fill_mode
    )
    
    #Load images 
    maize_train_image = load_img('./input_data/{}'.format(maize_images[0]))
    onion_train_image = load_img('./input_data/{}'.format(onions_images[0]))

    # Create a numpy array with shape (3, 150, 150)
    maize_train_image = img_to_array(maize_train_image)
    onion_train_image = img_to_array(onion_train_image)

    #Convert to Numpy array with shape (1, 3, 150, 150)
    maize_train_image = maize_train_image.reshape((1,) + maize_train_image.shape)
    onion_train_image = onion_train_image.reshape((1,) + onion_train_image.shape)

    #Generate more data samples 
    print('Generating more training data using Data Generator...')
         

    i = 1
    for batch in train_gen.flow(maize_train_image, save_to_dir = "./input_data", save_prefix = 'Maize', save_format = 'jpg'):
        i +=1
        if i > 10:
            break

    for batch in train_gen.flow(onion_train_image, save_to_dir = "./input_data", save_prefix = 'Onion', save_format = 'jpg'):
        i +=1
        if i > 10:
            break


    onions_and_maize_images = os.listdir('./input_data')      
    maize_images = list(filter(lambda image: 'Maize' in image, onions_and_maize_images))
    onions_images = list(filter(lambda image: 'Onion' in image, onions_and_maize_images))

    print(onions_and_maize_images)
    print(onions_images)
    print(maize_images)

    random.shuffle(onions_images)
    random.shuffle(maize_images)

    split_index = int(len(onions_images) * train_validation_split)
    
    training_maize = maize_images[:split_index]
    training_onions = onions_images[:split_index]    
    validation_maize = maize_images[split_index:]
    validation_onions = onions_images[split_index:]

    file_paths = [training_maize, training_onions,validation_maize, validation_onions]
    specific_path = ['train/maize', 'train/onions', 'validate/maize', 'validate/onions']

    for i in range(4):
        copy_images(file_paths[i], specific_path[i])


def train_model(train_iterator, validation_iterator):
    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=input_shape),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=metrics)

    print('Trainig the ML Model...')
    history = model.fit(train_iterator,
                        validation_data = validation_iterator,
                        steps_per_epoch = steps_per_epoch,
                        epochs = epochs,
                        validation_steps = validation_steps)

    model.save('maize-vs-onions.h5')

    return history


def plot_result(history):
    print('About to plot the results and graphs')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def load_and_predict():
    print('Loading the Model to perform inferences')
    model = keras.models.load_model('maize-vs-onions.h5')

    test_generator = ImageDataGenerator(rescale=1. / 255)

    test_iterator = test_generator.flow_from_directory(
        './input_test',
        target_size=(150, 150),
        shuffle=False,
        class_mode='binary',
        batch_size=1)

    ids = []
    for filename in test_iterator.filenames:
        ids.append(int(filename.split('\\')[1].split('.')[0]))

    predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
    predictions = []
    for index, prediction in enumerate(predict_result):
        predictions.append([ids[index], prediction[0]])
    predictions.sort()

    return predictions




create_folder_structure()
create_training_and_validation_set()

#result_history = train_model()
#result_history

#plot_result(result_history)

#predictions = load_and_predict()

#df = pd.DataFrame(data=predictions, index=range(1, 12501), columns=['id', 'label'])
#df = df.set_index(['id'])
#df.to_csv('submission.csv')

#create_folder_structure()
