
import numpy as np
import cv2

class ImageBinaryConverter:
    def __init__(self):
        pass

    def convert_image_to_binary(self, img):
        #use color transforms, gradients, etc., to create a thresholded binary image
        # cv2.imshow('Original', img)

        # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray', img)
        # lower = np.array([22, 93, 0], dtype="uint8")
        # upper = np.array([45, 255, 255], dtype="uint8")
        # mask = cv2.inRange(hls, lower, upper)

        # img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
        # cv2.i/mshow('Blur', blur)

        #
        #greather than second arg, set to 255, else 0
        ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Thrash', img)

        adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        cv2.imshow('Adaptive', adapt)


        low_threshold = 5
        high_threshold = 150    
        img = cv2.Canny(img, low_threshold, high_threshold)
        # cv2.imshow('Canny Edged', img)

        cv2.waitKey(0)