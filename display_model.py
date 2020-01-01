import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot

# load train and test dataset
def load_dataset():
	# load dataset from home/xavier/.keras/datasets/mnist.npz
	path = 'mnist.npz'
	(trainX, trainY), (testX, testY) = mnist.load_data(path)
	# reshape dataset to have a single channel, skip color
	#trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	#testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	#trainY = to_categorical(trainY)
	#testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# just to show pictures in the model
def show_data(trainX, trainY, testX, testY):
	# plot first few images
	for i in range(9):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# plot raw pixel data
		pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
	# show the figure
	pyplot.show()

# run_display
def run_display():

	#print("Tensorflow version: ", tf.__version__)
	# eager execution
	tf.executing_eagerly()
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# show data
	show_data(trainX, trainY, testX, testY)

#
#############################	
# entry point
#############################
run_display()
