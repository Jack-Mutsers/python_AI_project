
import cv2

def returnCameraIndexes():
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

def getCameras():
    cameras = returnCameraIndexes()

    items = []
    camera_nr = 0
    for camera in cameras:
        camera_nr += 1
        items.append("cam"+str(camera_nr))
    items.append("image")

    return (items,  cameras)