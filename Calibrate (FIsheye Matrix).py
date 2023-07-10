import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

CHECKERBOARD = (3, 3)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# path to the images
mypath = r'G:\AI Engineering\Co-ops\Chitwan Singh\Fisheye Distortion\Testing\Images\Calibration 25mm'
# Filter out folders and select only image files
image_files = [
    f
    for f in os.listdir(mypath)
    if os.path.isfile(os.path.join(mypath, f))
    and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
]

# Sort the image files numerically
images = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

for fname in images:
    print(fname)
    img = cv2.imread(mypath + "\\" + fname)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    print("Found corners in the image: " + str(ret))
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1)
    )


print("Found " + str(N_OK) + " valid images for calibration")
print("K = np.array(" + str(K.tolist()) + ")")
print("D = np.array(" + str(D.tolist()) + ")")

'''
object_points = []
image_points = []
pattern_points = np.zeros((np.prod(CHECKERBOARD), 3), dtype=np.float32)
pattern_points[:, :2] = np.indices(CHECKERBOARD).T.reshape(-1, 2)

# Find the corners of the chessboard pattern in the image
ret, corners = cv2.findChessboardCorners(img, CHECKERBOARD, None)

if ret:
    # Add the corners to the image and object points lists
    image_points.append(corners)
    object_points.append(pattern_points)

    # Calibrate the camera and obtain the distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img.shape[:2], None, None)
    
# Check for tangential distortion

print("Camera Matrix = np.array("  + str(mtx.tolist()) + ")")
print("Distortion Coefficient = np.array("  + str(dist.tolist()) + ")")


if abs(dist[0][3]) > 0.001 or abs(dist[0][2]) > 0.001:
    print("The image has tangential distortion.")
else:
    print("The image does not have tangential distortion.")'''



