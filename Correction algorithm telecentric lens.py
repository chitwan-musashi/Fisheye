import cv2
import numpy as np

# Step 2: Load the image
image = cv2.imread('G:\AI Engineering\Co-ops\Chitwan Singh\Fisheye Distortion\Testing\Images\Telecentric lens\End_face_exposure_20ms_01.jpg')

# Fisheye calibration parameters specific to your fisheye lens
K = np.array([[651.8986469044033, 0.0, 1023.5], [0.0, 651.8986469044033, 767.5], [0.0, 0.0, 1.0]])
D = np.array([[0.0], [0.0], [0.0], [0.0]])

undistorted_image = cv2.undistort(image, K, D)

# Display or save the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('undistorted_image.jpg', undistorted_image)
