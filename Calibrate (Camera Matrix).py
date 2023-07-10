import cv2
import numpy as np
import os

# Checkerboard settings
checkerboard_size = (6, 10)  # Number of inner corners (rows, columns)
square_size = 1.0  # Size of each square in the checkerboard (in any unit)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000, 0.1)

# Arrays to store object points and image points from all images
object_points_list = []  # 3D points in real-world space
image_points_list = []  # 2D points in image plane

# Path to the calibration images
calibration_images_folder = r'G:\AI Engineering\Co-ops\Chitwan Singh\Fisheye Distortion\Testing\Images\25 mm Samples'

# Filter out folders and select only image files
image_files = [
    f
    for f in os.listdir(calibration_images_folder)
    if os.path.isfile(os.path.join(calibration_images_folder, f))
    and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
]

# Process each image
for image_file in image_files:
    print(image_file)

    # Load the image
    image = cv2.imread(os.path.join(calibration_images_folder, image_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    print("Found CheckerBoard Corners: " + str(ret))

    # If found, add object points and image points
    if ret:
        object_points = np.zeros((np.prod(checkerboard_size), 1, 3), dtype=np.float32)
        object_points[:, 0, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        object_points *= square_size

        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
        object_points_list.append(object_points)
        image_points_list.append(corners_refined)

# Perform camera calibration using cv2.calibrateCamera
_, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
    object_points_list,
    image_points_list,
    gray.shape[::-1],
    None,
    None
)

'''# Perform camera calibration using cv2.fisheye.calibrate
calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_CHECK_COND
    + cv2.fisheye.CALIB_FIX_SKEW
)
N_OK = len(object_points_list)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]


rms, _, _, _, _ = cv2.fisheye.calibrate(
    object_points_list,
    image_points_list,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    criteria
)'''

# Print the camera matrices
print("Intrinsic Camera Matrix:")
print("K = np.array(" + str(camera_matrix.tolist()) + ")")
print("D = np.array(" + str(dist_coeffs.tolist()) + ")")

'''print("Fisheye Camera Matrix:")
print("K = np.array(" + str(K.tolist()) + ")")
print("D = np.array(" + str(D.tolist()) + ")")'''
