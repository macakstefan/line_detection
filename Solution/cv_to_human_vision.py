import cv2
import numpy as np
import matplotlib.pyplot as plt

class CVToHumanVision():
    def __init__(self, image_path, roi, new_dim):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.roi = roi
        self.new_dim = new_dim
        

    def draw_lines(self, poly_left, poly_right, img):
        # Generate y-values
        y_values = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_x_values = poly_left[0] * y_values**2 + poly_left[1] * y_values + poly_left[2]
        right_x_values = poly_right[0] * y_values**2 + poly_right[1] * y_values + poly_right[2]

        left_x_values = left_x_values.astype(int)
        right_x_values = right_x_values.astype(int)
        y_values = y_values.astype(int)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Draw the lines on the image
        for i in range(len(y_values) - 1):
            cv2.line(img, (left_x_values[i], y_values[i]), (left_x_values[i + 1], y_values[i + 1]), (255, 0, 0), 20)
            cv2.line(img, (right_x_values[i], y_values[i]), (right_x_values[i + 1], y_values[i + 1]), (0, 0, 255), 20)

        # Display the image
        cv2.imshow('Lanes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_lines_on_original_image(self, img, poly_left, poly_right):
    
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        y_values = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_x_values = poly_left[0] * y_values**2 + poly_left[1] * y_values + poly_left[2]
        right_x_values = poly_right[0] * y_values**2 + poly_right[1] * y_values + poly_right[2]

        left_x_values = left_x_values.astype(int)
        right_x_values = right_x_values.astype(int)
        y_values = y_values.astype(int)

        # Create a blank image for the polygon
        polygon_img = np.zeros_like(img)

        left_points = np.array([np.transpose(np.vstack([left_x_values, y_values]))])
        right_points = np.array([np.flipud(np.transpose(np.vstack([right_x_values, y_values])))])

        # Combine the points to form the polygon
        points = np.hstack((left_points, right_points))


        cv2.fillPoly(polygon_img, np.int_([points]), (255, 255, 0))

        M_inv = cv2.getPerspectiveTransform(self.new_dim, self.roi)
        warped_polygon = cv2.warpPerspective(polygon_img, M_inv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(self.image, 1, warped_polygon, 0.3, 0)

        # Display the image
        cv2.imshow('Lane Lines', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()