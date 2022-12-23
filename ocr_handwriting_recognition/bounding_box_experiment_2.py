# import the necessary packages
from keras.models import load_model
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

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

for image_path in image_paths:
    t=0