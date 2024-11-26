import os
import numpy as np
import pickle
import cv2
from calibration import ChessboardCameraCalibrator
from binary_converter import ImageBinaryConverter
import matplotlib.pyplot as plt
from lines_detector import LineDetector
from cv_to_human_vision import CVToHumanVision


print( os.getcwd())
image_path = "test_images/test3.jpg"
video_path = "C:\\Users\\macak\\Documents\\master\\computer vision\\Zadatak\\test_videos\\project_video02.mp4"

def do(img):
    calibrator = ChessboardCameraCalibrator(9,6)


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


    img = binary_converter.convert_image_to_binary(img)
    print("Binary image shape:", img.shape)

    #get img size 
    img_size = (img.shape[1], img.shape[0])
    img_height, img_width = img.shape[:2]
    #cut image to half and show image 
    # img = img[0:img_size[1]//2, 0:img_size[0]]
   

    roi = generate_roi(img)
    #write new dim based on picture size

    new_dim = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    M = cv2.getPerspectiveTransform(roi, new_dim)
    img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    cv2.imshow('Warped', img)
    try:
        ld = LineDetector(img, 20, 10, 10)
        poly_left, poly_right =ld.detect(img)
        converter = CVToHumanVision(image_path, roi, new_dim)
        converter.draw_lines_on_original_image(img, poly_left, poly_right)
    except Exception as e:
        print("Errroror:", e)
        
    # return 
        
    

def generate_roi(img):
    img_size = (img.shape[1], img.shape[0])
    img_height, img_width = img.shape[:2]
    # roi coordiinates 
    top_left = (img_size[0]//2 - 100, img_size[1]//2 + 100)
    top_right = (img_size[0]//2 + 100, img_size[1]//2 + 100)
    bottom_left = (20, img_size[1])
    bottom_right = (img_size[0], img_size[1])

    # Calculate ROI coordinates as a trapezoid on the left side of the image
    top_left = (int(img_width * 0.45), int(img_height * 0.62))
    top_right = (int(img_width * 0.55), int(img_height * 0.62))
    bottom_left = (int(img_width * 0.15), int(img_height * 0.9))
    bottom_right = (int(img_width * 0.95), int(img_height * 0.9))


    cv2.line(img, top_left, top_right, (0, 255, 0), 2)
    cv2.line(img, top_right, bottom_right, (0, 255, 0), 2)
    cv2.line(img, bottom_right, bottom_left, (0, 255, 0), 2)
    cv2.line(img, bottom_left, top_left, (0, 255, 0), 2)
    cv2.imshow('Image with ROI', img)
    # cv2.waitKey(0)

    return np.float32([top_left, top_right, bottom_left, bottom_right])



if __name__ == '__main__':
    img = cv2.imread(image_path)
    # roi(img)
    # do(img)

    video = cv2.VideoCapture(video_path)
    print("Video path exists:", os.path.exists(video_path))
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
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object and close all OpenCV windows
    video.release()
    cv2.destroyAllWindows()
