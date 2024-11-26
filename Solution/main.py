import os
import numpy as np
import pickle
import cv2
from calibration import ChessboardCameraCalibrator
from binary_converter import ImageBinaryConverter
import matplotlib.pyplot as plt
from lines_detector import LineDetector
from cv_to_human_vision import CVToHumanVision
from birds_perspective_converter import BirdsPerspectiveConverter


print("Current working dir:", os.getcwd())
IMAGE_PATH = "test_images/test3.jpg"
VIDEO_PATH = "C:\\Users\\macak\\Documents\\master\\computer vision\\Zadatak\\test_videos\\project_video02.mp4"
CALIBRATE = False
USE_IMAGE= True
DEBUG = True

def do(img, calibrate=False):
    original_img = img.copy()
    if calibrate:
        calibrator = ChessboardCameraCalibrator(9,6)

        #calibrate the camera
        #obtain camera matrix(mtx) and distortion coefficients(dist)
        mtx, dist = calibrator.calibrate("Zadatak/camera_cal")

        for img_name in os.listdir("Zadatak/camera_cal"):
            img_path = os.path.join("Zadatak/camera_cal", img_name)
            img = cv2.imread(img_path)
            calibrator.undistort_image(img, mtx, dist, os.path.join("Zadatak/Solution/output_undistorted_images", img_name))

    #use color transforms, gradients, etc., to create a thresholded binary image
    binary_converter = ImageBinaryConverter()


    img = binary_converter.convert_image_to_binary(img)
    print("Binary image shape:", img.shape)
    birds_perspective = BirdsPerspectiveConverter(img, DEBUG)
    img = birds_perspective.transform_to_birds_perspective()

   
    try:
        ld = LineDetector(img, 20, 10, 10)
        poly_left, poly_right =ld.detect(img)
        converter = CVToHumanVision(IMAGE_PATH, roi, new_dim)
        converter.draw_lines_on_original_image(img, poly_left, poly_right, original_img)
    except Exception as e:
        print("Errroror:", e)
        
    return 
        

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH)
    # roi(img)
    if USE_IMAGE:
        do(img)
        cv2.waitKey(0)

    else:
        video = cv2.VideoCapture(VIDEO_PATH)
        print("Video path exists:", os.path.exists(VIDEO_PATH))
        print(video.isOpened())

        # Process video frames
        while video.isOpened():
            ret, frame = video.read()
            
            if ret:
                do(frame)
                # generate_roi(frame)
                # Display the resulting frame
                # cv2.imshow('Frame', frame)

                # Add a delay and check for 'q' key to exit
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release the video capture object and close all OpenCV windows
        video.release()
        cv2.destroyAllWindows()
