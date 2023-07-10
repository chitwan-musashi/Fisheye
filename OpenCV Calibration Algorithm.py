import numpy as np
import cv2 as cv
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Checkercoard Dimensions
CHECKERBOARD = (6,3)
# Path to the calibration images
calibration_images_folder = r'G:\AI Engineering\Co-ops\Chitwan Singh\Fisheye Distortion\Testing\Angled Lights\Calibration Images 25 mm'

# Filter out folders and select only image files
images = [
    f
    for f in os.listdir(calibration_images_folder)
    if os.path.isfile(os.path.join(calibration_images_folder, f))
    and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
]


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.prod(CHECKERBOARD), 1, 3), dtype=np.float32)
objp[:, 0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


for fname in images:
    print(fname)
    img = cv.imread(os.path.join(calibration_images_folder, fname))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    print("Checkerboard Found: " + str(ret))
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

img = cv.imread(r'g:\AI Engineering\Co-ops\Chitwan Singh\Fisheye Distortion\Testing\Images\25 mm Samples\Image__2023-06-14__14-56-24.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult.png', dst)
cv.waitKey(0)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibresult.png', dst)
cv.waitKey(0)