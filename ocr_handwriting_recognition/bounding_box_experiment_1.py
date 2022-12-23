# https://pyimagesearch.com/2020/08/24/ocr_handwriting_recognition-with-opencv-keras-and-tensorflow/
# https://pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
# https://stackoverflow.com/questions/61297312/finding-the-bounding-boxes-of-different-symbols-letters

# import the necessary packages
import numpy as np
import cv2
import imutils

image_paths = [
    # r"ocr_handwriting_recognition/images/hello_world.png",
    # r"ocr_handwriting_recognition/images/img-01.png",
    # r"ocr_handwriting_recognition/images/img-02.jpeg", # upper with lowercase letters
    # r"ocr_handwriting_recognition/images/img-03.jpg", # handwritten fonts
    # r"ocr_handwriting_recognition/images/img-04.png", # names (white background)
    # r"ocr_handwriting_recognition/images/img-05.png", # uppercase only
    # r"ocr_handwriting_recognition/images/img-06.jpeg", # uper with lowercase letters (more space + no background)
    # r"ocr_handwriting_recognition/images/test_picture.PNG", # actual photo
    # r"ocr_handwriting_recognition/images/part-1.jpeg",
    # r"ocr_handwriting_recognition/images/part-2.jpeg",
    # r"ocr_handwriting_recognition/images/part-3.jpeg",
    # r"ocr_handwriting_recognition/images/because.png",
    # r"ocr_handwriting_recognition/images/letters.png",
    r"ocr_handwriting_recognition/images/multipart_characters.png",
]

def load_image(image_path):
    img=cv2.imread(image_path)
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

for image_path in image_paths:

    (image, imgGray, imgDilate) = load_image(image_path)

    (contours, hierarchy) = cv2.findContours(imgDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    (contours, _) = sort_contours(contours)

    (boxes, chars) = get_bounding_boxes(contours, imgGray)

    # for char in chars:
    #     cv2.imshow("Image", char)
    #     cv2.waitKey(0)

    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(_, boxes):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
