import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils
from PIL import Image
import numpy as np
import os


# To load images to features and labels
def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1, 28, 28, 1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data

# predict image
def predict_image(model):
    #
    print("Start predict image 1.png")
    img = Image.open('data/mnist_data/validation/1/1_2.png').convert("L")
    img = np.resize(img, (28,28,1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
    y_pred = model.predict_classes(im2arr)
    print(y_pred)


# normalize
def normalize(X_train, X_test):
    # normalize inputs from 0-255 to 0-1
    X_train/=255
    X_test/=255

# encode
def encode(number_of_classes, y_train, y_test):
    # one hot encode
    number_of_classes = 10
    y_train = to_categorical(y_train, number_of_classes)
    y_test = to_categorical(y_test, number_of_classes)

# create model()
def create_model(number_of_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model

# compile model
def compile_model(model):
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# fit model
def fit_model(model):
    # Fit the model
    model.fit(X_train, y_train, epochs=7, batch_size=200, validation_data=(X_test, y_test))
    #model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))
    metrics = model.evaluate(X_test, y_test, verbose=1)
    print("Metrics - (test loss and test accuracy)")
    print(metrics)
    return model

# save model
def save_model(model):
    # save model
	model.save('models/final_model2.h5')

# entry point, run training
def run_training():
    #
    number_of_classes = 10 # out classes
    normalize(X_train, X_test)
    encode(number_of_classes, y_train, y_test)
    model = create_model(number_of_classes)
    model = compile_model(model)
    #model = fit_model(model)
    save_model(model)

#############################	
# entry point
#############################
print("Start training data...")
# load data, Let’s load the Keras MNIST dataset first
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# Now let’s reshape the data according to CNN expectations
# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
#
X_train, y_train = load_images_to_data('1', 'data/mnist_data/train/1', X_train, y_train)
X_test, y_test = load_images_to_data('1', 'data/mnist_data/validation/1', X_test, y_test)
#
run_training()
print("End training...")