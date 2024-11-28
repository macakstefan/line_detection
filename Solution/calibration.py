import numpy as np
import cv2
import glob
import os

class ChessboardCameraCalibrator:
    def __init__(self, cb_cols, cb_rows, debug=False):
        self.cb_cols = cb_cols
        self.cb_rows = cb_rows
        self.objectPointsArray = []
        self.imgPointsArray = []
        self.debug = debug
        #matrix of zeros 2d (rows x cols) * 3
        self.objp = np.zeros((self.cb_rows*self.cb_cols,3), np.float32)
        #fill the first two columns with the coordinates of the chessboard corners
        self.objp[:,:2] = np.mgrid[0:self.cb_cols,0:self.cb_rows].T.reshape(-1,2)
        

    def get_calibration_params(self, path_with_calibration_images):
        chess_boards_images = glob.glob(path_with_calibration_images)        
        print("Number of images found:", len(chess_boards_images))

        for image_path in chess_boards_images:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if(self.debug):
                cv2.imshow('gray', gray)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.cb_rows, self.cb_cols))

            if ret == True:
                # Refine the corner position
                termination_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)

                self.objectPointsArray.append(self.objp)
                self.imgPointsArray.append(corners)
                if(self.debug):
                    # Draw the corners on the image
                    cv2.drawChessboardCorners(img, (self.cb_rows, self.cb_cols), corners, ret)
                    # Display the image
                    cv2.imshow(image_path, img)

            else:
                print("All Corners not found for image:", image_path)

            if self.debug:
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objectPointsArray, self.imgPointsArray, gray.shape[::-1], None, None)
        return  mtx, dist
    
    def undistort_image(self, calibration_data_path, img):
        if not os.path.exists(calibration_data_path):
            raise Exception("Calibration data file not found")
        
        calibration_data = np.load(calibration_data_path)

        mtx = calibration_data['mtx']
        dist = calibration_data['dist']
        h, w = img.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        calibrated_img =  cv2.undistort(img , mtx, dist, None, newCameraMtx)
        x, y, w, h = roi
        calibrated_img = calibrated_img[y:y+h, x:x+w]
        
        return calibrated_img


    def undistort_all_images_in_folder(self, calibration_data_path, input_images_path, output_images_path=""):
        

        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
        
        #check if input_images_path is a directory
        if not os.path.isdir(input_images_path):
            raise Exception("Input images path is not a directory")

        for img_name in os.listdir(input_images_path):
            img = cv2.imread(os.path.join(input_images_path, img_name))
            calibrated_img = self.undistort_image(calibration_data_path, img)
            output = os.path.join(output_images_path, img_name)
            cv2.imwrite(output, calibrated_img)
        
        return True
