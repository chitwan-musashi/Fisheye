import cv2
import numpy as np

# Step 2: Load the image
image = cv2.imread(r'g:\AI Engineering\Nyan\9 Feasibility Studies\05 GM Valve Body\Bench Testing\2D f75\Large Ring and Dome\f75_465mm-WD_f-num-16_30k-exp_RL_40mmWD_dome_250mmWD___2023-06-09_small_porosities.bmp')

# Fisheye calibration parameters specific to your fisheye lens
K = np.array([[24380.126099659654, 0.0, 1271.8609372473518], [0.0, 24633.907351012054, 568.949906578828], [0.0, 0.0, 1.0]])
D = np.array([[-21.133734368495368, 4490.966166867435, 0.01612968994427542, -0.29582926469564474, 0.014052812618222853]])


undistorted_image = cv2.undistort(image, K, D)

# Display or save the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('undistorted_f75_465mm-WD_f-num-16_30k-exp_RL_40mmWD_dome_250mmWD___2023-06-09_small_porosities.bmp', undistorted_image)
