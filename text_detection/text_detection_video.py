# https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

# import the necessary packages
from camera_controller import CameraController
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=False, help="path to input EAST text detector", default="text_detection/frozen_east_text_detection.pb")
ap.add_argument("-v", "--video", type=str, help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# start the app
pba = CameraController(args["east"], "output")
pba.root.mainloop()