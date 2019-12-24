# example of loading the mnist dataset
#import tensorflow as tf
#from tensorflow import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot


# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	#trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	#testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	#trainY = to_categorical(trainY)
	#testY = to_categorical(testY)
	return trainX, trainY, testX, testY


def show_data(trainX, trainY, testX, testY):
	# plot first few images
	for i in range(9):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# plot raw pixel data
		pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
	# show the figure
	pyplot.show()

def run_test_harness():

	#print("Tensorflow version: ", tf.__version__)
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# show data
	show_data(trainX, trainY, testX, testY)

# entry point
run_test_harness()