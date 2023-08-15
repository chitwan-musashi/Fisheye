import cv2
import numpy as np 

# Step 2: Load the image
image = cv2.imread(r'g:\AI Engineering\Co-ops\Chitwan Singh\Fisheye Distortion\Testing\Images\25 mm Samples\Image__2023-07-07__14-38-18.jpg')

# Fisheye calibration parameters specific to your fisheye lens
K = np.array([[25000, 0.0, 960], [0.0, 1500, 600], [0.0, 0.0, 1.0]])
D = np.array([[0.0], [0.0], [0.0], [0.0]])

# Step 4: Apply fisheye distortion correction
h, w = image.shape[:2]
map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
undistorted_fisheye_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Step 6: Display or save the corrected image
cv2.imshow('Undistorted Image', undistorted_fisheye_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('undistorted_image1_fx_25000__fy_1500__cx_980__cy_600.jpg', undistorted_fisheye_image)