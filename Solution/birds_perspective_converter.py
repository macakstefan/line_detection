import numpy as np
import cv2
import copy


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

         # Calculate the width and height of the trapezoid
        top_width = np.linalg.norm(np.array(roi[0]) - np.array(roi[1]))
        bottom_width = np.linalg.norm(np.array(roi[2]) - np.array(roi[3]))
        height = np.linalg.norm(np.array(roi[0]) - np.array(roi[2]))

        # Set new_dim to fit the entire trapezoid
        new_dim = np.float32([[0, 0],
                              [max(top_width, bottom_width), 0],
                              [0, height],
                              [max(top_width, bottom_width), height]])
        
        M = cv2.getPerspectiveTransform(roi, new_dim)
        result = cv2.warpPerspective(self.img, M, (int(max(top_width, bottom_width)), int(height)), flags=cv2.INTER_LINEAR)

        if(self.debug):
            cv2.imshow('Birds perspective', result)

        return result, roi, new_dim


    def generate_roi(self):
        # Calculate ROI coordinates as a trapezoid on the left side of the image
        top_left = (int(self.img_width * 0.40), int(self.img_height * 0.66))
        top_right = (int(self.img_width * 0.62), int(self.img_height * 0.66))
        bottom_left = (int(self.img_width * 0.07), int(self.img_height * 0.98))
        bottom_right = (int(self.img_width * 0.97), int(self.img_height * 0.98))

        #draw roi on image
        if(self.debug):
            roi_on_img = copy.deepcopy(self.img)
            roi_on_img = cv2.cvtColor(roi_on_img, cv2.COLOR_GRAY2RGB)
            cv2.line(roi_on_img, top_left, top_right, (0, 255, 0), 2)
            cv2.line(roi_on_img, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(roi_on_img, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(roi_on_img, bottom_left, top_left, (0, 255, 0), 2)
            
            cv2.imshow('Image with ROI', roi_on_img)

        return np.float32([top_left, top_right, bottom_left, bottom_right])
