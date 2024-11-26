
import cv2
import numpy as np
import matplotlib.pyplot as plt

class LineDetector:
    def __init__(self, img, window_size, window_stride, threshold):
        self.window_size = window_size
        self.window_stride = window_stride
        self.threshold = threshold
        self.img = img
        self.y = 0
        self.x = 1

    def calculate_histogram(self, img):
        histogram = np.sum(img, axis=0)
        # print(histogram)
        # plt.figure()
        # plt.plot(histogram)
        # plt.show()
        leftx_base = np.argmax(histogram[:len(histogram)//2])
        rightx_base = np.argmax(histogram[len(histogram)//2:]) + len(histogram)//2

        return leftx_base, rightx_base
        
        

    def detect(self, img):
        leftx_base, rightx_base = self.calculate_histogram(img)
        left_line_indexes = {"x": [], "y": []}
        right_line_indexes = {"x": [], "y": []}

        no_of_windows = 10 
        window_height = img.shape[0]//no_of_windows
        window_width = 75
        min_num_of_pixels_for_recentre = 5

        for (base, output) in [(leftx_base, left_line_indexes), (rightx_base, right_line_indexes)]:
            for i in range(no_of_windows):
                window = img.copy()
                window = window[window_height*i:window_height*(i+1), base-window_width:base+window_width]
                #use cv2 nonzeros to get all nonzero pixels in window
                nonzero = cv2.findNonZero(window)
                print("nonzero is:", nonzero)
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
                    print("nonzerox", nonzerox)
                    base = int(np.mean(nonzerox))
                    print("base", base)
                else:
                    print("nonzerox", len(nonzerox))
                    print("base", base)
                # cv2.rectangle(img, (base-window_width, window_height*i), (base+window_width, window_height*(i+1)), (255, 0, 0), 2)

        poly_left = np.polyfit(left_line_indexes["y"], left_line_indexes["x"], 2)
        poly_right = np.polyfit(right_line_indexes["y"], right_line_indexes["x"], 2)




        # Generate y-values
        y_values = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_x_values = poly_left[0] * y_values**2 + poly_left[1] * y_values + poly_left[2]
        right_x_values = poly_right[0] * y_values**2 + poly_right[1] * y_values + poly_right[2]

        left_x_values = left_x_values.astype(int)
        right_x_values = right_x_values.astype(int)
        y_values = y_values.astype(int)


        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)



        # # Draw the lines on the image
        for i in range(len(y_values) - 1):
            cv2.line(img, (left_x_values[i], y_values[i]), (left_x_values[i + 1], y_values[i + 1]), (255, 0, 0), 20)
            cv2.line(img, (right_x_values[i], y_values[i]), (right_x_values[i + 1], y_values[i + 1]), (0, 0, 255), 20)

        # Display the image
        cv2.imshow('Lanes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
     
                


        # cv2.imshow('Image', img)      
        # cv2.waitKey(0)     

        return []
    

    def write_on_original():
        

        
        polygon_img = np.zeros_like(img)

        # Define the points for the left and right lines
        left_points = np.array([np.transpose(np.vstack([left_x_values, y_values]))])
        right_points = np.array([np.flipud(np.transpose(np.vstack([right_x_values, y_values])))])

        # Combine the points to form the polygon
        points = np.hstack((left_points, right_points))

        top_left = (520,460)
        top_right = (740,460)
        bottom_left = (260,680)
        bottom_right = (1220,680)

        src = np.float32([top_left, top_right, bottom_left, bottom_right])
        dst = np.float32([[0,0], [1280,0], [0,720], [1280,720]])

        M_inv = cv2.getPerspectiveTransform(dst, src) 
        warped_polygon = cv2.warpPerspective(polygon_img, M_inv, (img.shape[1], img.shape[0]))

        cv2.fillPoly(polygon_img, np.int_([points]), (0, 255, 0))

        img = cv2.addWeighted(img, 1, warped_polygon, 0.3, 0)
