import tensorflow as tf
#from tensorflow import keras as keras
from tensorflow.keras.datasets import mnist
#import ssl
from matplotlib import pyplot

print("Tensorflow version: ", tf.__version__)

mnist = mnist.load_data(path="/home/xavier/Documents/pythonrep/src/tensorflow-demo/MNIST_data/t10k-images-idx-ubyte")

# load dataset

(trainX, trainY), (testX, testY) = mnist
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

# reshape dataset to have a single channel
