
# import the necessary packages
from __future__ import print_function
from types import NoneType
from imutils.video import VideoStream
from PIL import Image
from PIL import ImageTk
from imutils.object_detection import non_max_suppression
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import tkinter as tki
import threading
import datetime
import imutils
import time
import cv2
import os

class CameraController:
    def __init__(self, east, outputPath):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
        self.vs = None
        self.image = None
        self.outputPath = outputPath
        self.frame = None
        self.orig = None
        self.thread = None
        self.detection_thread = None
        self.stopEvent = None
        self.net = cv2.dnn.readNet(east)
        self.rW = None
        self.rH = None
        self.boxes = []

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        (self.items, self.cameras) = self.getCameras()

		# initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        self.combo = ttk.Combobox(self.root, state="readonly", values=self.items)
        self.combo.current(0)
        self.combo.bind('<<ComboboxSelected>>', self.choose)
        self.combo.pack(side="top", fill="both", expand="yes", padx=10, pady=10)

        self.choose()

		# create a button, that when pressed, will take the current
		# frame and save it to file
        btn = tki.Button(self.root, text="Snapshot!", command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

		# start a thread that constantly pools the video sensor for
		# the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        self.detection_thread = threading.Thread(target=self.detect_words, args=())
        self.detection_thread.start()

		# set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def decode_predictions(self, scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < 0.5:
                    continue
                
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

    def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
        try:
			# keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():

                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                if self.combo.get() == "image":
                    if type(self.image) == NoneType:
                        continue
                    self.frame = self.image
                else:
                    if type(self.vs) == NoneType:
                        continue
                    self.frame = self.vs.read()
                    
                self.frame = imutils.resize(self.frame, width=1000)
                self.orig = self.frame.copy()
        
                # loop over the bounding boxes
                for (startX, startY, endX, endY) in self.boxes:
                    # scale the bounding box coordinates based on the respective
                    # ratios
                    startX = int(startX * self.rW)
                    startY = int(startY * self.rH)
                    endX = int(endX * self.rW)
                    endY = int(endY * self.rH)

                    # draw the bounding box on the frame
                    cv2.rectangle(self.orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.orig, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
		
				# if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)
		
				# otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def detect_words(self):
        # initialize the original frame dimensions, new frame dimensions,
        # and ratio between the dimensions
        (W, H) = (None, None)
        (newW, newH) = (320, 320)

        while not self.stopEvent.is_set():
            if type(self.orig) == NoneType:
                continue

            if hasattr(self.orig, 'shape') is False:
                continue

            # if our frame dimensions are None, we still need to compute the
            # ratio of old frame dimensions to new frame dimensions
            if W is None or H is None:
                (H, W) = self.frame.shape[:2]
                self.rW = W / float(newW)
                self.rH = H / float(newH)

            # resize the frame, this time ignoring aspect ratio
            self.frame = cv2.resize(self.frame, (newW, newH))

            # construct a blob from the frame and then perform a forward pass
            # of the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(self.frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
            self.net.setInput(blob)
            (scores, geometry) = self.net.forward(self.layerNames)
            
            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = self.decode_predictions(scores, geometry)
            self.boxes = non_max_suppression(np.array(rects), probs=confidences, overlapThresh=0.1)

    def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))

		# save the file
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(filename))

    def returnCameraIndexes(self):
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        return arr

    def getCameras(self):
        cameras = self.returnCameraIndexes()

        items = []
        camera_nr = 0
        for camera in cameras:
            camera_nr += 1
            items.append("cam"+str(camera_nr))
        items.append("image")

        return (items,  cameras)

    def choose(self, event = None):
        camera_nr = 0
        for camera in self.cameras:
            camera_nr += 1
            if self.combo.get() == "cam"+str(camera_nr):
                print("[INFO] starting video stream...")
                self.image = None
                self.vs = VideoStream(src=camera).start()
                time.sleep(1.0)

        if self.combo.get() == "image":
            if type(self.vs) != NoneType:
                self.vs.stop()
            self.vs = None
            path = filedialog.askopenfilename( title="Select file", filetypes=(('JPG','*.jpg'), ('JPEG','*.jpeg'),('PNG', '*.png')))
            self.image = cv2.imread(path)

    def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()