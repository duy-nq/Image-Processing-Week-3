import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read johnnybin.sec
with open('johnnybin.sec', 'rb') as f:
    johnny = np.fromfile(f, dtype=np.uint8)

# Reshape johnny to 256x256
johnny = johnny.reshape((256, 256))

# Equalize histogram of johnny
johnny_hist = cv.equalizeHist(johnny)

# Plot johnny and histogram for original and modified johnny
plt.subplot(121), plt.imshow(johnny_hist, cmap='gray')
plt.title('Modified johnny'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.hist(johnny_hist.ravel(), 256, [0, 256])
plt.title('Histogram for modified johnny')
plt.show()
