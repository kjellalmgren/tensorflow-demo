import tensorflow as tf
from numpy import mean
from numpy import std
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils import np_utils
from PIL import Image
from matplotlib import pyplot
import numpy as np
import os


# To load images to features and labels
def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            print("processing {0:s}".format(image_file_name))
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,28,28,1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
            #print(features_data)
            #print(label_data)
    return features_data, label_data

# predict image
def predict_image(model):
    #
    print("Start predict image 1-1.png")
    img = Image.open('data/mnist_data/validation/1/1-1.png').convert("L")
    img = np.resize(img, (28,28,1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
    #im2arr = im2arr.reshape(28,28,1)
    y_pred = model.predict_classes(im2arr)
    print(y_pred)


# create model()
def create_model(number_of_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten(input_shape=(28, 28, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))
    return model

# compile model
def compile_model(model):
    # Compile model
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer="rmsprop",  metrics=['accuracy'])
    return model

# fit model
def fit_model(model):
    # Fit the model
    scores, histories = list(), list()
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    #model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY))
    metrics = model.evaluate(X_test, y_test, verbose=0)
    print("Metrics - (test loss and test accuracy)")
    print(metrics)
    # evaluate model
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
	# stores scores
    scores.append(acc)
    histories.append(history)
    return model, scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# save model
def save_model(model):
    # save model
	model.save('models/final_model1.h5')

# entry point, run training
def run_training():
    #
    number_of_classes = 10 # out classes
    model = create_model(number_of_classes)
    model = compile_model(model)
    model, scores, histories = fit_model(model)
    summarize_diagnostics(histories)
	# summarize estimated performance
    summarize_performance(scores)
    #
    save_model(model)
    #predict_image(model)

#############################	
# entry point
#############################
print("Start training data...")
number_of_classes = 10 # out classes
# load data, Let’s load the Keras MNIST dataset first
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# Now let’s reshape the data according to CNN expectations
# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
#
X_train, y_train = load_images_to_data('0', 'data/mnist_data/train/0', X_train, y_train)
X_test, y_test = load_images_to_data('0', 'data/mnist_data/validation/0', X_test, y_test)
#
X_train, y_train = load_images_to_data('1', 'data/mnist_data/train/1', X_train, y_train)
X_test, y_test = load_images_to_data('1', 'data/mnist_data/validation/1', X_test, y_test)
#
X_train, y_train = load_images_to_data('2', 'data/mnist_data/train/2', X_train, y_train)
X_test, y_test = load_images_to_data('2', 'data/mnist_data/validation/2', X_test, y_test)
#
X_train, y_train = load_images_to_data('3', 'data/mnist_data/train/3', X_train, y_train)
X_test, y_test = load_images_to_data('3', 'data/mnist_data/validation/3', X_test, y_test)
#
X_train, y_train = load_images_to_data('4', 'data/mnist_data/train/4', X_train, y_train)
X_test, y_test = load_images_to_data('4', 'data/mnist_data/validation/4', X_test, y_test)
#
X_train, y_train = load_images_to_data('5', 'data/mnist_data/train/5', X_train, y_train)
X_test, y_test = load_images_to_data('5', 'data/mnist_data/validation/5', X_test, y_test)
#
X_train, y_train = load_images_to_data('6', 'data/mnist_data/train/6', X_train, y_train)
X_test, y_test = load_images_to_data('6', 'data/mnist_data/validation/6', X_test, y_test)
#
X_train, y_train = load_images_to_data('7', 'data/mnist_data/train/7', X_train, y_train)
X_test, y_test = load_images_to_data('7', 'data/mnist_data/validation/7', X_test, y_test)
#
X_train, y_train = load_images_to_data('8', 'data/mnist_data/train/8', X_train, y_train)
X_test, y_test = load_images_to_data('8', 'data/mnist_data/validation/8', X_test, y_test)
#
X_train, y_train = load_images_to_data('9', 'data/mnist_data/train/9', X_train, y_train)
X_test, y_test = load_images_to_data('9', 'data/mnist_data/validation/9', X_test, y_test)
#
# normalize
X_train/=255
X_test/=255
#
#encode
y_train = to_categorical(y_train, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)
#
run_training()
print("End training...")