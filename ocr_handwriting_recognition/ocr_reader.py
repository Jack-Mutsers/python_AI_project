# https://pyimagesearch.com/2020/08/24/ocr_handwriting_recognition-with-opencv-keras-and-tensorflow/
# https://pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
# https://stackoverflow.com/questions/61297312/finding-the-bounding-boxes-of-different-symbols-letters

# import the necessary packages
from keras.models import load_model
import numpy as np
import cv2
import imutils
from spellchecker import SpellChecker

# model_path = r"models/back-up/2022-11-10_11-11-16/handwriting-lowercase-4x100.model"
model_path = r"models/new/handwriting_perfect_letters_v2.model"
# model_path = r"models/working/handwriting_lowercase_uppercase_clasification.model"

# define the list of label names
labelNames = ""
# labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames += "abcdefghijklmnopqrstuvwxyz"
labelNames = [l for l in labelNames]

spell = SpellChecker()

# load the handwriting OCR model
print("[INFO] loading handwriting OCR model...")
model = load_model(model_path)

def load_image(image):
    img = None
    if isinstance(image, str):
        img=cv2.imread(image)
    else:
        img = image

    copy=img.copy()

    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    (_, imgThresh) = cv2.threshold(imgGray,70,255,cv2.THRESH_BINARY_INV)

    kernel_erosion = np.ones((1,1),np.uint8)
    imgErode = cv2.erode(imgThresh,kernel_erosion,iterations = 2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,2))
    imgOpen = cv2.morphologyEx(imgErode, cv2.MORPH_OPEN, kernel_open)

    kernel_dilate = np.ones((2,2),np.uint8)
    imgDilate = cv2.dilate(imgOpen,kernel_dilate,iterations = 4)

    return copy, imgGray, imgDilate

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_bounding_boxes(contours, imgGray):
    boxes=[] #store the dimensions of my bounding boxes
    chars=[] #store each bounding box for later
    min_cont_area=5
    
    for i, cnt in enumerate(contours):
        #print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > min_cont_area: #Limit the contours based on area?  
            (x, y, w, h) = cv2.boundingRect(cnt)
            
            if y > 3:
                y -= 3
            if x > 3:
                x -= 3

            boxes.append([x,y,w,h]) 
            
            roi=imgGray[y:y+h, x:x+w]

            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            chars.append(padded)
    
    chars = np.array([c for c in chars], dtype="float32")

    return boxes, chars

def read_image(frame):
    # cv2.imshow("Image", frame)
    # cv2.waitKey(0)

    (image, imgGray, imgDilate) = load_image(frame)

    (contours, hierarchy) = cv2.findContours(imgDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        return "", ""

    (contours, _) = sort_contours(contours)

    (boxes, chars) = get_bounding_boxes(contours, imgGray)

    # for char in chars:
    #     cv2.imshow("Image", char)
    #     cv2.waitKey(0)

    if len(chars) < 1:
        return "", ""

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)

    prediction = ""
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        
        prediction += label

    # print("prediction: "+prediction)
    expectation = ""
    if spell.candidates(prediction) != None:
        # Get the one `most likely` answer
        expectation = spell.correction(prediction)
    else:
        expectation = "[error] word not found"
    
    # print("expected: "+expectation)

    return prediction, expectation