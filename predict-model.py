# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
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
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	#img = img.reshape(1, 28, 28, 1)
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

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')
	#
#
def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array, true_label[i], img[i]

	plt.grid(True)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

  	#plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
  	#                              100*np.max(predictions_array),
  	#                              class_names[true_label]),
  	#                              color=color)
	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
		100*np.max(predictions_array),
		class_names[true_label]),
		color=color)
# load an image and predict the class
def run_code():

	
	# load the image
	img = load_image('images/seven.png')
	img1 = load_raw_image('images/seven.png')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predictions = model.predict_classes(img)
	digit = model.predict(img)
	print("prediction is calculated to : ", predictions[0])
	print(digit)
    #
	i = 0
	plt.figure(figsize=(6,3))
	plt.subplot(1,2,1)
	plot_image(i, digit, test_labels, img1)
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
