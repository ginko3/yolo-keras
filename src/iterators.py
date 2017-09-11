import os

import numpy as np
import cv2

def frameIt(folder="images"):
    for image_path in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, image_path)

        # Skip if not image
        if not image_path.endswith(".jpg"):
            continue

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if (yield image):
            break

def cameraIt(cam_id=0):
    capture = cv2.VideoCapture(0)
    b_continue = True
    while b_continue:
        ret, frame = capture.read()
        it_receive = (yield frame)

        # If frame is read correctly and no stop signal is received
        b_continue = ret and (not it_receive)
