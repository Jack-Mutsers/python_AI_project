# https://pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/
# start command: python ocr_keras_tensorflow/train_ocr_model.py --az ocr_keras_tensorflow/a_z_handwritten_data.csv --model ocr_keras_tensorflow/handwriting.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.models import load_model
from pyimagesearch.models.resnet import ResNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
from os.path import exists
import pyimagesearch.dataset.helpers as helpers
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt
import numpy as np
import argparse
import reporter
import shutil
import cv2
import sys
import os
from termcolor import colored

# emnist_letters_dataset_path = r"ocr_keras_tensorflow/pyimagesearch/dataset/emnist/emnist_letters_train_lowercase.csv"
emnist_letters_dataset_path = r"ocr_keras_tensorflow/pyimagesearch/dataset/emnist/emnist_letters_train_trimmed_letters_only.csv"
# emnist_dataset_path = r"ocr_keras_tensorflow/pyimagesearch/dataset/emnist/emnist_byclass_train_modded.csv"
# emnist_dataset_path = r"ocr_keras_tensorflow/pyimagesearch/dataset/emnist/emnist_byclass_train.csv"
emnist_dataset_path = r"ocr_keras_tensorflow/pyimagesearch/dataset/emnist/emnist_byclass_train_trimmed_letters_only.csv"
custom_dataset_path = r"ocr_keras_tensorflow\\pyimagesearch\dataset\\letter_e.csv"
AZ_dataset_path = r"ocr_keras_tensorflow/pyimagesearch/dataset/a_z_handwritten_data.csv"

perfect_letters = r"ocr_keras_tensorflow\\pyimagesearch\dataset\\perfect_letters_sm.csv"
perfect_joined_letters = r"ocr_keras_tensorflow\\pyimagesearch\dataset\\perfect_joined_letters_sm.csv"
typed_letters = r"ocr_keras_tensorflow\\pyimagesearch\dataset\\typed_letters_sm.csv"

model_path = r"models/new/handwriting_perfect_letters_v2.model"
# model_path = r"models/new/test.model"

add_delay = False
make_backup = False
copy_model = True
loaded_datasets = []
continuation = exists(model_path)

# initialize the number of epochs to train for, initial learning rate,
# and batch size
TRAIN_SESSIONS = 2
EPOCHS = 50
INIT_LR = 8e-3
BS = 800  #batch size

if os.path.exists("models") is False:
	os.makedirs("models")
	os.makedirs("models/back-up")
	os.makedirs("models/new")
	os.makedirs("models/incorrect")
	os.makedirs("models/working")

if add_delay:
	import time

	current_time = dt.datetime.now()
	await_time = dt.datetime.now() + dt.timedelta(hours=0, minutes=30)
	print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))
	print("starting time: " + await_time.strftime("%Y-%m-%d %H:%M:%S"))

	while(current_time < await_time):
		time.sleep(310)
		current_time = dt.datetime.now()
		print("waiting to start")
		print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))

	print("done waiting")

if make_backup:
	dst_dir=r"models/new/handwriting-lowercase-back-up.model"
	shutil.copy(model_path,dst_dir)

# optimise memory chunking for graphical prosessing cards
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
SGD = tf.keras.optimizers.SGD

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print(os.getenv("TF_GPU_ALLOCATOR"))

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.experimental.set_virtual_device_configuration(gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])


start_time = dt.datetime.now()
print("run started at: " + start_time.strftime("%Y-%m-%d %H:%M:%S"))

# define the list of label names
# labelNames = "0123456789"
labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames += "abcdefghijklmnopqrstuvwxyz"
labelNames = [l for l in labelNames]

# load the A-Z and MNIST datasets, respectively
print("[INFO] loading datasets...")
# (azData, azLabels) = helpers.load_az_dataset(AZ_dataset_path)
# (digitsData, digitsLabels) = helpers.load_mnist_dataset()
(emnistClassData, emnistClassLabels) = helpers.load_az_dataset(emnist_dataset_path, flipped=True)
(emnistLettersData, emnistLettersLabels) = helpers.load_az_dataset(emnist_letters_dataset_path, flipped=True)
(customData, customLabels) = helpers.load_az_dataset(custom_dataset_path, flipped=False)
(perfectLettersData, perfectLettersLabels) = helpers.load_az_dataset(perfect_letters, flipped=False)
(perfectJoinedLettersData, perfectJoinedLettersLabels) = helpers.load_az_dataset(perfect_joined_letters, flipped=False)
(typedLettersData, typedLettersLabels) = helpers.load_az_dataset(typed_letters, flipped=False)
print("[INFO] datasets loaded.")

# label_set1 = set(emnistClassLabels)
# label_set2 = set(emnistLettersLabels)

# the MNIST dataset occupies the labels 0-9, so let's add 10 to every
# A-Z label to ensure the A-Z characters are not incorrectly labeled as digits
# azLabels += 10
# emnistLettersLabels += 9 # +9 because it starts at index 1 instead of 0
emnistClassLabels -= 10 # -10 because the numbers have been removed
emnistLettersLabels -= 10 # -10 because the numbers have been removed
customLabels -= 10
perfectLettersLabels -= 10
perfectJoinedLettersLabels -= 10
typedLettersLabels -= 10

# stack the A-Z data and labels with the MNIST digits data and labels

data = np.empty([0,28,28], "uint8")
labels = np.empty([0,])

# loaded_datasets.append(AZ_dataset_path)
# data = np.vstack([data, azData])
# labels = np.hstack([labels, azLabels])

# loaded_datasets.append("mnist_dataset")
# data = np.vstack([data, digitsData])
# labels = np.hstack([labels, data, digitsLabels])

loaded_datasets.append(emnist_letters_dataset_path)
data = np.vstack([data, emnistLettersData])
labels = np.hstack([labels, emnistLettersLabels])

loaded_datasets.append(emnist_dataset_path)
data = np.vstack([data, emnistClassData])
labels = np.hstack([labels, emnistClassLabels])

loaded_datasets.append(custom_dataset_path)
data = np.vstack([data, customData])
labels = np.hstack([labels, customLabels])

load_times = 30
loaded_datasets.append(perfect_letters + " (x" +load_times + ")")
loaded_datasets.append(perfect_joined_letters + " (x" +load_times + ")")
loaded_datasets.append(typed_letters + " (x" +load_times + ")")
for i in range(0, load_times):
	data = np.vstack([data, perfectLettersData])
	labels = np.hstack([labels, perfectLettersLabels])

	data = np.vstack([data, perfectJoinedLettersData])
	labels = np.hstack([labels, perfectJoinedLettersLabels])

	data = np.vstack([data, typedLettersData])
	labels = np.hstack([labels, typedLettersLabels])


labels_set = set(labels)
if(len(labels_set) != len(labelNames)):
	print()
	print(labels_set)
	print()
	print(labelNames)
	print()
	sys.exit(colored("ValueError: Number of classes, "+ str(len(labels_set)) +", does not match the size of the labelNames, "+ str(len(labelNames)) +". Try specifying the labels parameter", "red"))

for i in range(TRAIN_SESSIONS):
	print("started training session " + str(i+1))

	print("shuffle dataset records")
	(newData, newLabels, classWeight, le) = helpers.shuffle(data, labels, 3*i)

	# partition the data into training and testing splits using 80% of
	# the data for training and the remaining 10% for testing
	# X == data, Y == labels
	(trainX, testX, trainY, testY) = train_test_split(newData, newLabels, test_size=0.10, stratify=newLabels, random_state=7*i)

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(
		rotation_range=10,
		zoom_range=0.05,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.15,
		horizontal_flip=False,
		fill_mode="nearest")

	if(continuation == False):
		# initialize and compile our deep neural network
		print("[INFO] compiling model...")
		# opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
		opt = SGD(learning_rate=INIT_LR, decay=0.001)
		model = ResNet.build(width=32, height=32, depth=1, classes=len(le.classes_), stages=(3, 4, 4), filters=(64, 64, 128, 256), reg=0.0005)
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	else:
		print("[INFO] loading existing model...")
		model = load_model(model_path)

	# train the network
	print("[INFO] training network...")

	training_time = dt.datetime.now()
	print("training started at: " + training_time.strftime("%Y-%m-%d %H:%M:%S"))

	batch_size = BS
	if len(trainX) < BS:
		batch_size = 1 

	H = model.fit(
		aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) / batch_size,
		epochs=EPOCHS,
		class_weight=classWeight,
		verbose=1
	)

	# save the model to disk
	print("[INFO] serializing network...")
	model.save(model_path, save_format="h5")

	if copy_model:
		model_path_arr = model_path.split("/")
		new_model_path = "models/back-up/" + start_time.strftime("%Y-%m-%d_%H-%M-%S") + "/"
		
		if os.path.exists(new_model_path) is False:
			os.makedirs(new_model_path)

		model_name = model_path_arr[-1].split(".")
		new_model_name = new_model_path + model_name[0] + "-" + str(i+1) + "x" + str(EPOCHS) + "." + model_name[1]
		shutil.copy(model_path, new_model_name)
		reporter.log_training_settings(INIT_LR, batch_size, EPOCHS, i, loaded_datasets, continuation, new_model_path)


	# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=batch_size)
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
		filepath = "ocr_keras_tensorflow/plots/" + start_time.strftime("%Y-%m-%d_%H-%M-%S") + "/"

		if os.path.exists(filepath) is False:
			os.makedirs(filepath)

		filename = filepath + helpers.get_unique_filename("plot")
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
		cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
		# cv2.putText(image, labelNames[np.argmax(testY[i])], (88, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
		cv2.putText(img=image, text=labelNames[np.argmax(testY[i])], org=(70, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(180, 180, 0), thickness=2)

		# add the image to our list of output images
		images.append(image)

	# construct the montage for the images
	montage = build_montages(images, (96, 96), (7, 7))[0]

	tf.keras.backend.clear_session()

	# show the output montage
	try:
		filepath = "ocr_keras_tensorflow/images/" + start_time.strftime("%Y-%m-%d_%H-%M-%S") + "/"

		if os.path.exists(filepath) is False:
			os.makedirs(filepath)

		filename = filepath + helpers.get_unique_filename("OCR Results")
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