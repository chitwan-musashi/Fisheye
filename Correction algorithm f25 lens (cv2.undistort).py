import cv2
import numpy as np

# Step 2: Load the image
distorted_image = cv2.imread(r"g:\AI Engineering\Co-ops\Chitwan Singh\Fisheye Distortion\Testing\Images\25 mm Samples\Image__2023-07-07__14-38-18.jpg")

# Fisheye calibration parameters specific to your fisheye lens
K = np.array([[30000, 0.0, 971.2720921280946], [0.0, 9000, 605.5997197100847], [0.0, 0.0, 1.0]])
D = np.array([[-15.597942932291792, 1612.890499576513, 0, 0, 9.315227517850253]])

# Step 4: Calculate the optimal camera matrix
image_size = (distorted_image.shape[1], distorted_image.shape[0])
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, image_size, 1)

# Step 5: Undistort the image
undistorted_image = cv2.undistort(distorted_image, K, D)

# Step 6: Display or save the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('undistorted_image2_fx_20000__fy_1500__cx_980__cy_600.jpg', undistorted_image)