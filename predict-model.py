# make a prediction for a new image.
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as pyimg
import numpy as np


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
#
def load_raw_image(filename):
	# load the image
	img = pyimg.imread(filename)
	#plt.imshow(pyimg.imread(filename))
	#img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	#print(img)
	# reshape into a single sample with 1 channel
	#img = img.reshape(1, 28, 28)
	# prepare pixel data
	#img = img.astype('float32')
	#img = img / 255.0
	return img
#
# plot_value_array
def plot_value_array(i, predictions_array, true_label): 

	predictions_array, true_label = predictions_array, true_label[i]
	plt.grid(False)
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('green')
	thisplot[true_label].set_color('blue')
	plt.show()
	#
#
def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array, true_label[i], img[i]

	#plt.grid(True)
	plt.xticks([])
	plt.yticks([])
	#
	#plt.subplot(230 + 1)
	#img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	#img = img.astype('float32')
	#img = img / 255.0
	# cmap=pyplot.get_cmap('gray')
	# cmap=plt.cm.binary
	#plt.imshow(img, cmap=plt.cm.binary)
	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'red'
	else:
		color = 'green'

  	#plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
  	#                              100*np.max(predictions_array),
  	#                              class_names[true_label]),
  	#                              color=color)
	plt.xlabel("{} {:4.2f}% ({})".format(class_names[predicted_label],
		100*np.max(predictions_array),
		class_names[true_label]),
		color=color)
#
# load an image and predict the class
def run_code():

	# load the image
	img = load_image('images/two-5.png')
	img1 = load_raw_image('images/two-5.png')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predictions = model.predict_classes(img)
	digit = model.predict(img)
	print("prediction is calculated to : {0:0d} ({1:1s} {2:4.2f}%)".format(predictions[0], class_names[predictions[0]], 100*np.max(digit)))
	print("prediction is calculated to : {0:0d}".format(predictions[0]))
	print("prediction is calculated to : {0:1s}".format(class_names[predictions[0]]))
	print("prediction is calculated to : {0:4.2f}%".format(100*np.max(digit)))
	#print("Array:", digit)
	#for names in class_names:
	#	print("name: %s" % (names))
	#
	j = 0
	sum = 0
	#print("list %s" % (predictions[0].tolist()))
	items = digit[0].tolist()
	print(items)
	for item in items:
		print("index: {0:2d} - {1:10.8f} - {2:10.8%}".format(j, item, (item)))
		j = j + 1
		sum = sum + item
	#
	print("Sum: {:3.2f}".format(sum))
	i = 0
	plt.figure(figsize=(6,3))
	plt.subplot(1,2,1)
	plot_image(i, digit, test_labels, img1)
	plt.imshow(img1)
	plt.subplot(1,2,2)
	plot_value_array(i, digit[0], test_labels)
	plt.show()
	#
#
# entry point
class_names = ['zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine']
test_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
run_code()
