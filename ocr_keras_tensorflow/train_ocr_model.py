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
import helpers.dataset_loader as dataset_loader
import helpers.reporter as reporter
import shutil
import cv2
import sys
import os
from termcolor import colored

class OCR_TRAINER:

	def __init__(self):
		self.model_path = r"models/new/handwriting_perfect_letters.model"
		# self.model_path = r"models/new/test.model"

		self.model_back_up = True
		self.loaded_datasets = []
		self.continuation = exists(self.model_path)

		# define the list of label names
		# labelNames = "0123456789"
		labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		labelNames += "abcdefghijklmnopqrstuvwxyz"
		self.labelNames = [l for l in labelNames]

		# initialize the number of epochs to train for, initial learning rate,
		# and batch size
		self.train_sessions = 1
		self.epochs = 1
		self.INIT_LR = 5e-2
		self.BS = 800  #batch size
		self.start_time = None
		self.ImageDataGenerator = None
		self.SGD = None

		if os.path.exists("models") is False:
			os.makedirs("models")
			os.makedirs("models/back-up")
			os.makedirs("models/new")
			os.makedirs("models/working")
		
		self.gpu_optimizer()

	def delay(self, hours = 0, minutes = 30, seconds = 0):
		print("")
		print("initiated delay")
		import time

		current_time = dt.datetime.now()
		await_time = dt.datetime.now() + dt.timedelta(hours = hours, minutes = minutes, seconds = seconds)
		print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))
		print("starting time: " + await_time.strftime("%Y-%m-%d %H:%M:%S"))

		while(current_time < await_time):
			time.sleep(310)
			current_time = dt.datetime.now()
			print("waiting to start")
			print("current time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))

		print("done waiting")
		print("")

	def gpu_optimizer(self):
		# optimise memory chunking for graphical prosessing cards
		self.ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
		self.SGD = tf.keras.optimizers.SGD

		os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
		print(os.getenv("TF_GPU_ALLOCATOR"))

		gpus = tf.config.experimental.list_physical_devices("GPU")
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
			tf.config.experimental.set_virtual_device_configuration(gpu,
				[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])

	def load_datasets(self):
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

		load_AZ = {
			"location": AZ_dataset_path, "offset": 0, "amount": 0
		}

		datasets_to_load = [
			{"location": emnist_letters_dataset_path, 	"flipped": True,  "offset": -10, "amount": 1},
			{"location": emnist_dataset_path, 			"flipped": True,  "offset": -10, "amount": 1},
			{"location": custom_dataset_path, 			"flipped": False, "offset": -10, "amount": 1},
			{"location": perfect_letters, 				"flipped": False, "offset": -10, "amount": 30},
			{"location": perfect_joined_letters, 		"flipped": False, "offset": -10, "amount": 30},
			{"location": typed_letters, 				"flipped": False, "offset": -10, "amount": 30},
		]

		return dataset_loader.load_dataset(
			load_mnist = False, 
			load_AZ = load_AZ, 
			datasets_to_load = datasets_to_load
		)

	def validate_loaded_dataset(self, labels):
		labels_set = set(labels)
		if(len(labels_set) != len(self.labelNames)):
			print()
			print(labels_set)
			print()
			print(self.labelNames)
			print()
			sys.exit(colored("ValueError: Number of classes, "+ str(len(labels_set)) +", does not match the size of the labelNames, "+ str(len(self.labelNames)) +". Try specifying the labels parameter", "red"))

	def backup_model(self, batch_size, session):
		model_path_arr = self.model_path.split("/")
		model_name = model_path_arr[-1].split(".")

		paths = [
			model_name[0],
			self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
		]

		new_model_path = "models/back-up/"
		for path in paths:
			new_model_path += (path + "/")
			
			if os.path.exists(new_model_path) is False:
				os.makedirs(new_model_path)

		new_model_name = new_model_path + model_name[0] + "-" + str(session) + "x" + str(self.epochs) + "." + model_name[1]
		shutil.copy(self.model_path, new_model_name)
		reporter.log_training_settings(self.INIT_LR, batch_size, self.epochs, session, self.loaded_datasets, self.continuation, new_model_path)

	def create_box_plot(self, history):
		# construct a plot that plots and saves the training history
		N = np.arange(0, self.epochs)
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(N, history.history["loss"], label="train_loss")
		plt.plot(N, history.history["val_loss"], label="val_loss")
		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		try:
			filepath = "ocr_keras_tensorflow/training_results/plots/" + self.start_time.strftime("%Y-%m-%d_%H-%M-%S") + "/"

			if os.path.exists(filepath) is False:
				os.makedirs(filepath)

			filename = filepath + helpers.get_unique_filename("plot")
			plt.savefig(filename)
		except Exception as e:
			print("could not save plot")

	def create_training_showcase(self, testData, testLabels, model):
		# initialize our list of output test images
		images = []

		# randomly select a few testing characters
		for i in np.random.choice(np.arange(0, len(testLabels)), size=(49,)):
			# classify the character
			probs = model.predict(testData[np.newaxis, i])
			prediction = probs.argmax(axis=1)
			label = self.labelNames[prediction[0]]

			# extract the image from the test data and initialize the text
			# label color as green (correct)
			image = (testData[i] * 255).astype("uint8")
			color = (0, 255, 0)

			# otherwise, the class label prediction is incorrect
			if prediction[0] != np.argmax(testLabels[i]):
				color = (0, 0, 255)

			# merge the channels into one image, resize the image from 32x32
			# to 96x96 so we can better see it and then draw the predicted
			# label on the image
			image = cv2.merge([image] * 3)
			image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
			cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
			# cv2.putText(image, labelNames[np.argmax(testLabels[i])], (88, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
			cv2.putText(img=image, text=self.labelNames[np.argmax(testLabels[i])], org=(70, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(180, 180, 0), thickness=2)

			# add the image to our list of output images
			images.append(image)

		# construct the montage for the images
		montage = build_montages(images, (96, 96), (7, 7))[0]

		# cv2.imshow("OCR Results", montage)
		# cv2.waitKey(0)

		# show the output montage
		try:
			filepath = "ocr_keras_tensorflow/training_results/images/" + self.start_time.strftime("%Y-%m-%d_%H-%M-%S") + "/"

			if os.path.exists(filepath) is False:
				os.makedirs(filepath)

			filename = filepath + helpers.get_unique_filename("OCR Results")
			cv2.imwrite(filename, montage)
		except Exception as e:
			print("could not save image")

	def load_model(self, classes):
		if(self.continuation == False):
			# initialize and compile our deep neural network
			print("[INFO] compiling model...")
			# opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / epochs)
			opt = self.SGD(learning_rate=self.INIT_LR, decay=0.001)
			model = ResNet.build(width=32, height=32, depth=1, classes=classes, stages=(3, 4, 4), filters=(64, 64, 128, 256), reg=0.0005)
			model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
		else:
			print("[INFO] loading existing model...")
			model = load_model(self.model_path)

		return model

	def evaluate_model(self, model, testData, testLabels, batch_size):
		# evaluate the network
		print("[INFO] evaluating network...")
		predictions = model.predict(testData, batch_size=batch_size)
		print(classification_report(testLabels.argmax(axis=1), predictions.argmax(axis=1), target_names=self.labelNames))

	def train_model(self, model, training_setings, session):
		# partition the data into training and testing splits using 80% of
		# the data for training and the remaining 10% for testing
		# X == data, Y == labels
		(trainData, testData, trainLabels, testLabels) = train_test_split(training_setings["data"], training_setings["labels"], test_size=0.10, stratify=training_setings["labels"], random_state=7*session)

		# construct the image generator for data augmentation
		aug = self.ImageDataGenerator(
			rotation_range=10,
			zoom_range=0.05,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=0.15,
			horizontal_flip=False,
			fill_mode="nearest")
		
		# train the network
		print("[INFO] training network...")

		training_time = dt.datetime.now()
		print("training started at: " + training_time.strftime("%Y-%m-%d %H:%M:%S"))

		batch_size = self.BS
		if len(trainData) < batch_size:
			batch_size = len(trainData) 

		H = model.fit(
			aug.flow(trainData, trainLabels, batch_size=batch_size),
			validation_data=(testData, testLabels),
			steps_per_epoch=len(trainData) / batch_size,
			epochs=self.epochs,
			class_weight=training_setings["classWeight"],
			verbose=1
		)

		# save the model to disk
		print("[INFO] serializing network...")
		model.save(self.model_path, save_format="h5")

		if self.model_back_up:
			self.backup_model(batch_size, session)

		self.evaluate_model(model, testData, testLabels, batch_size)

		return model, H, {"data": testData, "labels": testLabels}

	def main(self):
		self.start_time = dt.datetime.now()
		print("run started at: " + self.start_time.strftime("%Y-%m-%d %H:%M:%S"))

		(data, labels, self.loaded_datasets) = self.load_datasets()
		self.validate_loaded_dataset(labels)

		for i in range(self.train_sessions):
			print("started training session " + str(i+1))

			print("shuffle dataset records")
			(newData, newLabels, classWeight, classes) = helpers.shuffle(data, labels, 3*i)

			model = self.load_model(classes)

			training_setings = {
				"data": newData,
				"labels": newLabels,
				"classWeight": classWeight
			}

			(model, history, testDataset) = self.train_model(model, training_setings, i+1)

			self.create_box_plot(history)
			self.create_training_showcase(testDataset["data"], testDataset["labels"], model)

			tf.keras.backend.clear_session()

		end_time = dt.datetime.now()
		duration = end_time-self.start_time
		print("run finished at: " + end_time.strftime("%Y-%m-%d %H:%M:%S"))
		print("total duration: " + str(duration))



if __name__ == "__main__":
	trainer = OCR_TRAINER()
	trainer.main()
