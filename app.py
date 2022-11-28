import ssl
import cv2
from PIL import ImageGrab
from dnsdetection import chart_recognisation
import numpy as np


def ml_detection_input():
    ssl._create_default_https_context = ssl._create_unverified_context

    while True:
        screen = np.array(ImageGrab.grab())
        frame = np.uint8(screen)
        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
        ret = 1

        if frame is None:
            break
        chart_recognisation(ret, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


ml_detection_input()
