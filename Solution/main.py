import os
import numpy as np
import cv2
from calibration import ChessboardCameraCalibrator
from binary_converter import ImageBinaryConverter
from lines_detector import LineDetector
from cv_to_human_vision import CVToHumanVision
from birds_perspective_converter import BirdsPerspectiveConverter


print("Current working dir:", os.getcwd())
IMAGE_PATH = "test_images/test1.jpg"
VIDEO_PATH = "C:\\Users\\macak\\Documents\\master\\computer vision\\Zadatak\\test_videos\\project_video01.mp4"
# CALIBRATE = True
CALIBRATE = False
USE_IMAGE= True
USE_IMAGE= False
DEBUG = True

def do(img, calibrate=False):
    calibrator = ChessboardCameraCalibrator(9,6, DEBUG)

    if calibrate:

        #obtain camera matrix(mtx) and distortion coefficients(dist)
        mtx, dist = calibrator.get_calibration_params("camera_cal/*.jpg")
        np.savez('Solution/calibration_data.npz', mtx=mtx, dist=dist)

        calibrator.undistort_all_images_in_folder('Solution/calibration_data.npz',
                                    'camera_cal',
                                    'Solution/output_undistorted_images')

    
    original_img = calibrator.undistort_image('Solution/calibration_data.npz', img)
    # img = calibrator.undistort_image('Solution/calibration_data.npz', img)

    # return
    original_img = img.copy()

    birds_perspective = BirdsPerspectiveConverter(img,USE_IMAGE, DEBUG)
    img, roi, new_dim, M = birds_perspective.transform_to_birds_perspective()
    # return


    #use color transforms, gradients, etc., to create a thresholded binary image
    binary_converter = ImageBinaryConverter()

    img = binary_converter.convert_image_to_binary(img)
    print("Binary image shape:", img.shape)


   
    try:
        ld = LineDetector(img, 20, 10, 10, DEBUG)
        poly_left, poly_right =ld.detect()  
        converter = CVToHumanVision(roi, new_dim, DEBUG)
        converter.draw_lines(poly_left, poly_right, img)
        result = converter.draw_lines_on_original_image(img, poly_left, poly_right, original_img, M)

    except Exception as e:
        print("Errroror:", e)
        
        

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH)
    # roi(img)
    if USE_IMAGE:
        do(img, CALIBRATE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        video = cv2.VideoCapture(VIDEO_PATH)
        print("Video path exists:", os.path.exists(VIDEO_PATH))
        print(video.isOpened())

        # Process video frames
        while video.isOpened():
            ret, frame = video.read()
            
            if ret:
                do(frame, CALIBRATE)

                # Add a delay and check for 'q' key to exit
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release the video capture object and close all OpenCV windows
        video.release()
        cv2.destroyAllWindows()
