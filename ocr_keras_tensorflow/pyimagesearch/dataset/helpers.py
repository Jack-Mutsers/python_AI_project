# import the necessary packages
from keras.datasets import mnist
import numpy as np
import uuid
import datetime
import random
import cv2
from imutils import build_montages
from sklearn.preprocessing import LabelBinarizer

def reshape_and_rotate(image):
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

def load_az_dataset(datasetPath, flipped = False):
	# initialize the list of data and labels
	data = []
	labels = []

	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):
		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")

		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))

		if flipped:
			image = reshape_and_rotate(image)

		# update the list of data and labels
		data.append(image)
		labels.append(label)

	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")

	# return a 2-tuple of the A-Z data and labels
	return (data, labels)

def load_mnist_dataset():
	# load the MNIST dataset and stack the training data and testing
	# data together (we'll create our own training and testing splits
	# later in the project)
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
	
	# return a 2-tuple of the MNIST data and labels
	return (data, labels)

def get_unique_filename(filename):
	unique_filename = str(uuid.uuid4())
	currentDT = datetime.datetime.now()
	return filename + "_" + currentDT.strftime("%Y-%m-%d_%H-%M-%S") + ".png"

def shuffle(data, labels, amount):

	tmpArr = []
	for i in range(0, len(data)):
		tmpArr.append([data[i], labels[i]])

	for i in range(0, amount):
		random.shuffle(tmpArr)
	
	newData = []
	newLabels = []
	for record in tmpArr:
		newData.append(record[0])
		newLabels.append(record[1])

	return prepare_data(newData, newLabels)

def prepare_data(data, labels):
	# each image in the A-Z and MNIST digts datasets are 28x28 pixels;
	# however, the architecture we're using is designed for 32x32 images,
	# so we need to resize them to 32x32
	data = [cv2.resize(image, (32, 32)) for image in data]
	data = np.array(data, dtype="float32")

	# add a channel dimension to every image in the dataset and scale the
	# pixel intensities of the images from [0, 255] down to [0, 1]
	data = np.expand_dims(data, axis=-1)
	data /= 255.0

	# convert the labels from integers to vectors
	le = LabelBinarizer()
	labels = le.fit_transform(labels)

	# account for skew in the labeled data
	classTotals = labels.sum(axis=0)
	classWeight = {}

	# loop over all classes and calculate the class weight
	for i in range(0, len(classTotals)):
		classWeight[i] = classTotals.max() / classTotals[i]

	return (data, labels, classWeight, len(le.classes_))

def show_loaded_dataset(labelNames, data, labels):
	images = []

	# randomly select a few testing characters
	for i in range(0,49):
		# classify the character
		label = labelNames[labels[i]]

		# extract the image from the test data and initialize the text
		# label color as green (correct)
		image = (data[i] * 255).astype("uint8")
		color = (0, 255, 0)

		# merge the channels into one image, resize the image from 32x32
		# to 96x96 so we can better see it and then draw the predicted
		# label on the image
		image = cv2.merge([image] * 3)
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
		cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
		# cv2.putText(image, labelNames[np.argmax(testY[i])], (88, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
		# cv2.putText(img=image, text=labelNames[np.argmax(labels[i])], org=(70, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(180, 180, 0), thickness=2)

		# add the image to our list of output images
		images.append(image)
	
	# construct the montage for the images
	montage = build_montages(images, (96, 96), (7, 7))[0]
	cv2.imshow("OCR Results", montage)
	cv2.waitKey(0)