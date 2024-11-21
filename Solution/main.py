import os
import pickle
import cv2
from calibration import ChessboardCameraCalibrator
from binary_converter import ImageBinaryConverter


if __name__ == '__main__':
    calibrator = ChessboardCameraCalibrator(9,6)

    print( os.getcwd())

    #calibrate the camera
    #obtain camera matrix(mtx) and distortion coefficients(dist)
    # mtx, dist = calibrator.calibrate("Zadatak/camera_cal")

    with open('calibration.p', mode='wb') as f:
        mtx, dist = calibrator.get_calibration_params("Zadatak/camera_cal/*.jpg")
        pickle.dump([ mtx, dist], f)
        f.close()
    
  
    '''
    for img_name in os.listdir("Zadatak/camera_cal"):
        img_path = os.path.join("Zadatak/camera_cal", img_name)
        img = cv2.imread(img_path)
        calibrator.undistort_image(img, mtx, dist, os.path.join("Zadatak/Solution/output_undistorted_images", img_name))
    '''
    #use color transforms, gradients, etc., to create a thresholded binary image
    binary_converter = ImageBinaryConverter()
    img = cv2.imread("Zadatak/test_images/test1.jpg")
    binary_converter.convert_image_to_binary(img)
   
