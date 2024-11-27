
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

class LineDetector:
    def __init__(self, img, window_size, window_stride, threshold, debug=False):
        self.window_size = window_size
        self.window_stride = window_stride
        self.threshold = threshold
        self.img = img
        self.y = 0
        self.x = 1
        self.debug = debug

    def calculate_histogram(self, img):
        histogram = np.sum(img, axis=0)
        # if self.debug:
        #     plt.figure()
        #     plt.plot(histogram)
        #     plt.show()   

        return np.argmax(histogram)
       

    def detect(self):
        img = copy.deepcopy(self.img)
        left_line_indexes = {"x": [], "y": []}
        right_line_indexes = {"x": [], "y": []}



        no_of_windows = 10 
        window_height = img.shape[0]//no_of_windows
        window_width = 75
        min_num_of_pixels_for_recentre = 5

        
        #callc histogram of window.

        left_window = img[:img.shape[0], :img.shape[1]//2]
        leftx_base = self.calculate_histogram(left_window)
        right_window = img[:img.shape[0], img.shape[1]//2:]
        rightx_base = self.calculate_histogram(right_window) + img.shape[1]//2

        for (base, output) in [(leftx_base, left_line_indexes), (rightx_base, right_line_indexes)]:
            for i in range(no_of_windows):
                window = img.copy()
                # Adjust base if it goes out of bounds
                if base - window_width < 0:
                    base = window_width
                elif base + window_width > img.shape[1]:
                    base = img.shape[1] - window_width

                window_x_border = (base-window_width, base+window_width)
                window = window[window_height*i:window_height*(i+1), window_x_border[0]:window_x_border[1]]
                #use cv2 nonzeros to get all nonzero pixels in window
                nonzero = cv2.findNonZero(window)
                # print("nonzero is:", nonzero)
                # nonzero = window.nonzero()
                
                # nonzerox = np.array(nonzero[self.x])
                if nonzero is None:
                    nonzerox = []
                else:
                    nonzerox = np.array([x[0][0] for x in nonzero]) + (base - window_width)
                    nonzeroy = np.array([x[0][1] for x in nonzero]) + (window_height * i)
                    output["x"].extend(nonzerox)
                    output["y"].extend(nonzeroy)           

                if len(nonzerox) > min_num_of_pixels_for_recentre:
                    # print("nonzerox", nonzerox)
                    base = int(np.mean(nonzerox))
                    # print("base", base)
                else:
                    # print("nonzerox", len(nonzerox))
                    print("base", base)

                if self.debug:
                    rectangle = cv2.rectangle(img, (base-window_width, window_height*i), (base+window_width, window_height*(i+1)), (255, 0, 0), 2)
                    cv2.imshow('Rectangle', rectangle)

                
        poly_left = np.polyfit(left_line_indexes["y"], left_line_indexes["x"], 2)
        poly_right = np.polyfit(right_line_indexes["y"], right_line_indexes["x"], 2)


        return poly_left, poly_right
    
