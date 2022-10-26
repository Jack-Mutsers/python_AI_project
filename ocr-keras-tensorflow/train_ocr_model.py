# https://pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/
# start command: python ocr-keras-tensorflow/train_ocr_model.py --az ocr-keras-tensorflow/a_z_handwritten_data.csv --model ocr-keras-tensorflow/handwriting.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.models import load_model
from pyimagesearch.models.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
from os.path import exists
import pyimagesearch.dataset.helpers as helpers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import datetime as dt
import sys
import os
from termcolor import colored

emnist_letters_dataset_path = r"ocr-keras-tensorflow/pyimagesearch/dataset/emnist/emnist-letters-train-modded.csv"
# emnist_dataset_path = r"ocr-keras-tensorflow/pyimagesearch/dataset/emnist/emnist-byclass-train-modded.csv"
emnist_dataset_path = r"ocr-keras-tensorflow/pyimagesearch/dataset/emnist/emnist-byclass-train.csv"
AZ_dataset_path = r"ocr-keras-tensorflow/pyimagesearch/dataset/a_z_handwritten_data.csv"
model_path = r"models/new/handwriting-lowercase.model"

add_delay = False
if add_delay:
	import time

	current_time = dt.datetime.now()
	await_time = dt.datetime.now() + dt.timedelta(hours=1, minutes=57)
	print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))
	print("run time: " + await_time.strftime("%Y-%m-%d %H:%M:%S"))

	while(current_time < await_time):
		time.sleep(1800)
		current_time = dt.datetime.now()
		print("waiting to start")
		print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))

	print("done waiting")


	import shutil
	dst_dir=r"models/new/handwriting-lowercase-back-up.model"
	shutil.copy(model_path,dst_dir)

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
SGD = tf.keras.optimizers.SGD

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=False, help="path to A-Z dataset", default=AZ_dataset_path)
ap.add_argument("-e", "--emnist", required=False, help="path to emnist dataset", default=emnist_dataset_path)
ap.add_argument("-m", "--model", type=str, required=False, help="path to output trained handwriting recognition model", default=model_path)
ap.add_argument("-cn", "--createnew", type=bool, required=False, help="Overwrite curently existing model if one exists on the specified model path", default=False)
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 40
INIT_LR = 1e-1
BS = 100  #batch size

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print(os.getenv("TF_GPU_ALLOCATOR"))

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

start_time = dt.datetime.now()
print("run started at: " + start_time.strftime("%Y-%m-%d %H:%M:%S"))

# define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames += "abcdefghijklmnopqrstuvwxyz"
labelNames = [l for l in labelNames]

# load the A-Z and MNIST datasets, respectively
print("[INFO] loading datasets...")
(azData, azLabels) = helpers.load_az_dataset(args["az"])
(digitsData, digitsLabels) = helpers.load_mnist_dataset()
(emnistClassData, emnistClassLabels) = helpers.load_az_dataset(args["emnist"], flipped=True)
(emnistLettersData, emnistLettersLabels) = helpers.load_az_dataset(emnist_letters_dataset_path, flipped=True)
print("[INFO] datasets loaded.")

# the MNIST dataset occupies the labels 0-9, so let's add 10 to every
# A-Z label to ensure the A-Z characters are not incorrectly labeled as digits
azLabels += 10
emnistLettersLabels += 9 # +9 because it starts at 1 instead of 0

# stack the A-Z data and labels with the MNIST digits data and labels

# 0:00 hours per 10 epochs
data = np.vstack([azData, digitsData, emnistLettersData])
labels = np.hstack([azLabels, digitsLabels, emnistLettersLabels])

# 3:10:29 hours per 10 epochs
# data = np.vstack([azData, digitsData, emnistLettersData, emnistClassData])
# labels = np.hstack([azLabels, digitsLabels, emnistLettersLabels, emnistClassLabels])

# 1:20:00 hours per 10 epochs
# data = np.vstack([emnistClassData]) 	
# labels = np.hstack([emnistClassLabels])

labels_set = set(labels)
if(len(labels_set) != len(labelNames)):
	sys.exit(colored("ValueError: Number of classes, 37, does not match size of target_names, 36. Try specifying the labels parameter", "red"))

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
counts = labels.sum(axis=0)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

train_sessions = 2
for i in range(train_sessions):
	# log_path = "ocr-keras-tensorflow/logs/" + start_time.strftime("%Y-%m-%d %H_%M")
	# Path(log_path).mkdir(parents=True, exist_ok=True)

	# f = open(log_path + "/training_session_"+ str(i) +".md", 'w')
	# sys.stdout = f

	print("started training session " + str(i+1))

	# partition the data into training and testing splits using 80% of
	# the data for training and the remaining 20% for testing
	(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(
		rotation_range=10,
		zoom_range=0.05,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.15,
		horizontal_flip=False,
		fill_mode="nearest")

	condition_1 = exists(args["model"])
	condition_2 = args["createnew"]

	if(exists(args["model"]) == False or args["createnew"]):
		# initialize and compile our deep neural network
		print("[INFO] compiling model...")
		opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
		model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3), (64, 64, 128, 256), reg=0.0005)
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	else:
		print("[INFO] loading existing model...")
		model = load_model(args["model"])

	# train the network
	print("[INFO] training network...")

	training_time = dt.datetime.now()
	print("training started at: " + training_time.strftime("%Y-%m-%d %H:%M:%S"))

	H = model.fit(
		aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS,
		class_weight=classWeight,
		verbose=1)

	# save the model to disk
	print("[INFO] serializing network...")
	model.save(args["model"], save_format="h5")

	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=BS)
	print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

	# construct a plot that plots and saves the training history
	N = np.arange(0, EPOCHS)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	try:
		filename = "ocr-keras-tensorflow/plots/" + helpers.get_unique_filename("plot")
		plt.savefig(filename)
	except Exception as e:
		print("could not save plot")

	# initialize our list of output test images
	images = []

	# randomly select a few testing characters
	for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
		# classify the character
		probs = model.predict(testX[np.newaxis, i])
		prediction = probs.argmax(axis=1)
		label = labelNames[prediction[0]]

		# extract the image from the test data and initialize the text
		# label color as green (correct)
		image = (testX[i] * 255).astype("uint8")
		color = (0, 255, 0)

		# otherwise, the class label prediction is incorrect
		if prediction[0] != np.argmax(testY[i]):
			color = (0, 0, 255)

		# merge the channels into one image, resize the image from 32x32
		# to 96x96 so we can better see it and then draw the predicted
		# label on the image
		image = cv2.merge([image] * 3)
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
		cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
			color, 2)

		# add the image to our list of output images
		images.append(image)

	# construct the montage for the images
	montage = build_montages(images, (96, 96), (7, 7))[0]


	# show the output montage
	try:
		filename = "ocr-keras-tensorflow/images/" + helpers.get_unique_filename("OCR Results")
		cv2.imwrite(filename, montage)
	except Exception as e:
		print("could not save image")

	# f.close()

end_time = dt.datetime.now()
duration = end_time-start_time
print("run finished at: " + end_time.strftime("%Y-%m-%d %H:%M:%S"))
print("total duration: " + str(duration))

# cv2.imshow("OCR Results", montage)
# cv2.waitKey(0)