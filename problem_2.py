import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read ladybin.sec
with open('ladybin.sec', 'rb') as f:
    lady = np.fromfile(f, dtype=np.uint8)

lady = lady.reshape((256, 256))

# Full-scale contrast stretching
lady_equ = cv.equalizeHist(lady)

# Sublot the lady_equ and histogram
plt.subplot(121)
plt.imshow(lady_equ, cmap='gray')
plt.title('lady_equ')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.hist(lady_equ.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.show()

# Wait for the user to press a key
cv.waitKey(0)
cv.destroyAllWindows()
