import cv2 as cv
import numpy as np

# Load the .bin file
with open('mammogrambin.sec', 'rb') as f:
    mammo = np.fromfile(f, dtype=np.uint8)

with open('Mammogram (1)bin.sec', 'rb') as f:
    mammo1 = np.fromfile(f, dtype=np.uint8)

# Reshape the image into a 2D array
mammo1 = mammo1.reshape((256, 256))

# Funtion for simple thresholding
def simple_thresholding(img, threshold):
    # Create a new image
    new_img = np.zeros((256, 256), dtype=np.uint8)
    # Apply thresholding
    new_img[img > threshold] = 255
    return new_img

new_img = simple_thresholding(mammo1, 100)

cv.imshow('Simple Thresholding', new_img)

# Function for approximate contour image generation
def approximate_contour(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (255, 255, 255), 1)
    return img

new_img = approximate_contour(new_img)

# Display the image
cv.imshow('ACIG', new_img)

# Wait for the user to press a key
cv.waitKey(0)
cv.destroyAllWindows()