import numpy as np
import cv2


class BirdsPerspectiveConverter:
    def __init__(self, img, debug=False):
        self.img = img
        self.img_size = (img.shape[1], img.shape[0])
        self.img_height, self.img_width = img.shape[:2]
        self.debug = debug

    def transform_to_birds_perspective(self):
        result = None
        roi = self.generate_roi()
        #write new dim based on picture size
        new_dim = np.float32([[0, 0],
                                    [self.img_width, 0],
                                    [0, self.img_height],
                                    [self.img_width, self.img_height]])
        
        M = cv2.getPerspectiveTransform(roi, new_dim)
        result = cv2.warpPerspective(self.img, M, self.img_size, flags=cv2.INTER_LINEAR)

        if(self.debug):
            cv2.imshow('Birds perspective', result)

        return result


    def generate_roi(self):
        # Calculate ROI coordinates as a trapezoid on the left side of the image
        top_left = (int(self.img_width * 0.45), int(self.img_height * 0.62))
        top_right = (int(self.img_width * 0.55), int(self.img_height * 0.62))
        bottom_left = (int(self.img_width * 0.15), int(self.img_height * 0.9))
        bottom_right = (int(self.img_width * 0.95), int(self.img_height * 0.9))

        #draw roi on image
        if(self.debug):
            roi_on_img = self.img.copy()
            roi_on_img = cv2.cvtColor(roi_on_img, cv2.COLOR_GRAY2RGB)
            cv2.line(roi_on_img, top_left, top_right, (0, 255, 0), 2)
            cv2.line(roi_on_img, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(roi_on_img, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(roi_on_img, bottom_left, top_left, (0, 255, 0), 2)
            
            cv2.imshow('Image with ROI', roi_on_img)

        return np.float32([top_left, top_right, bottom_left, bottom_right])