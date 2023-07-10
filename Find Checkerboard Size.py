import cv2
import numpy as np

# Load the input image
image = cv2.imread(r'Images/Calibration 25mm/Image__2023-07-10__10-07-35.jpg')


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Parameters for corner detection
min_rows = 3  # Minimum number of rows
min_cols = 3  # Minimum number of columns
max_rows = 11 # Maximum number of rows
max_cols = 11 # Maximum number of columns

# Variables to store best board size and accuracy
best_board_rows = None
best_board_cols = None
best_board_accuracy = 0.0

# Iterate through possible board sizes
for rows in range(min_rows, max_rows + 1):
    for cols in range(min_cols, max_cols + 1):

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If corners are found, calculate accuracy and draw the corners
        if ret:
            print("Corners found at: " + str(rows) + " Rows X " + str(cols) + " Columns")
            # Refine the corners to sub-pixel accuracy
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            )
            cv2.drawChessboardCorners(gray, (cols, rows), corners, ret)
            cv2.imshow('img', gray)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

            # Calculate accuracy as the percentage of detected corners
            accuracy = len(corners) / (rows * cols)

            # Update the best board size if accuracy is higher
            if accuracy > best_board_accuracy:
                best_board_accuracy = accuracy
                best_board_rows = rows
                best_board_cols = cols

            # Draw the corners on the image
            cv2.drawChessboardCorners(image, (cols, rows), corners, ret)

# If a board size is found, print the detected board size and display the image
if best_board_rows is not None and best_board_cols is not None:
    # Print the detected board size
    print("Best detected checkerboard size:", best_board_cols, "columns x", best_board_rows, "rows")

    # Display the image with the detected corners
    cv2.imshow("Detected Checkerboard", image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    #cv2.imwrite('undistorted_f75_465mm-WD_f-num-16_30k-exp_RL_40mmWD_dome_250mmWD___2023-06-09_small_porosities.bmp', image)
else:
    print("Checkerboard not found in the image.")

print("DONE!")