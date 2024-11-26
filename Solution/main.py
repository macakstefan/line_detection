import os
import numpy as np
import pickle
import cv2
from calibration import ChessboardCameraCalibrator
from binary_converter import ImageBinaryConverter
import matplotlib.pyplot as plt
from lines_detector import LineDetector
from cv_to_human_vision import CVToHumanVision

image_path = "test_images/test2.jpg"

if __name__ == '__main__':
    calibrator = ChessboardCameraCalibrator(9,6)

    print( os.getcwd())

    #calibrate the camera
    #obtain camera matrix(mtx) and distortion coefficients(dist)
    # mtx, dist = calibrator.calibrate("Zadatak/camera_cal")
  
    '''
    for img_name in os.listdir("Zadatak/camera_cal"):
        img_path = os.path.join("Zadatak/camera_cal", img_name)
        img = cv2.imread(img_path)
        calibrator.undistort_image(img, mtx, dist, os.path.join("Zadatak/Solution/output_undistorted_images", img_name))
    '''
    #use color transforms, gradients, etc., to create a thresholded binary image
    binary_converter = ImageBinaryConverter()
    img = cv2.imread(image_path)
    img = binary_converter.convert_image_to_binary(img)

    #get img size 
    img_size = (img.shape[1], img.shape[0])
    #cut image to half and show image 
    # img = img[0:img_size[1]//2, 0:img_size[0]]

    # roi coordiinates 
    top_left = (img_size[0]//2 - 100, img_size[1]//2 + 100)
    top_right = (img_size[0]//2 + 100, img_size[1]//2 + 100)
    bottom_left = (20, img_size[1])
    bottom_right = (img_size[0], img_size[1])

    top_left = (520,460)
    top_right = (740,460)
    bottom_left = (260,680)
    bottom_right = (1220,680)

    roi = np.float32([top_left, top_right, bottom_left, bottom_right])
    new_dim = np.float32([[0,0], [1280,0], [0,720], [1280,720]])
    M = cv2.getPerspectiveTransform(roi, new_dim)
    img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)


    ld = LineDetector(img, 20, 10, 10)
    poly_left, poly_right =ld.detect(img)
    converter = CVToHumanVision(image_path, roi, new_dim)
    converter.draw_lines_on_original_image(img, poly_left, poly_right)


