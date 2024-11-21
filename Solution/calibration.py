import numpy as np
import cv2
import glob

class ChessboardCameraCalibrator:
    def __init__(self, cb_cols, cb_rows):
        self.cb_cols = cb_cols
        self.cb_rows = cb_rows
        self.objectPointsArray = []
        self.imgPointsArray = []
        #matrix of zeros 2d (rows x cols) * 3
        self.objp = np.zeros((self.cb_rows*self.cb_cols,3), np.float32)
        #fill the first two columns with the coordinates of the chessboard corners
        self.objp[:,:2] = np.mgrid[0:self.cb_cols,0:self.cb_rows].T.reshape(-1,2)
        

    def get_calibration_params(self, path_with_calibration_images):
        chess_boards_images = glob.glob(path_with_calibration_images)        

        for image_path in chess_boards_images:
            # print("Image path is:", image_path)
            img = cv2.imread(image_path)
            # img = cv2.bilateralFilter(img, 9, 75, 75)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('gray', gray)
            # cv2.waitKey(2000)
            

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.cb_rows, self.cb_cols))

            # if corners is not None:
                # print("number of corners detected:", len(corners))
             #if corners are found, append object points and image points
            if ret == True:
                # Refine the corner position
                termination_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)

                self.objectPointsArray.append(self.objp)
                self.imgPointsArray.append(corners)
                 # Draw the corners on the image
                # cv2.drawChessboardCorners(img, (self.cb_rows, self.cb_cols), corners, ret)
                # Display the image
                # cv2.imshow(image_path, img)
                # cv2.waitKey(0)
            else:
                print("All Corners not found for image:", image_path)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objectPointsArray, self.imgPointsArray, gray.shape[::-1], None, None)
        return  mtx, dist
    

    def undistort_image(self, img, mtx, dist, output_image_path=""):
        calibrated_img =  cv2.undistort(img , mtx, dist, None, mtx)
        if output_image_path != "":
            cv2.imwrite(output_image_path, calibrated_img)
        return calibrated_img