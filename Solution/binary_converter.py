
import cv2

class ImageBinaryConverter:
    def __init__(self, debug=False):
        self.debug = debug

    def convert_image_to_binary(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)

        ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)

        low_threshold = 5
        high_threshold = 150    
        img = cv2.Canny(img, low_threshold, high_threshold)
        if(self.debug):
            cv2.imshow('Binary image', img)
        
        return img
