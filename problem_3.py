import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read actontbin.sec
with open('actontbin.sec', 'rb') as f:
    actont = np.fromfile(f, dtype=np.uint8)

actont = actont.reshape((256, 256))

# Binary Template Matching for letter 'T' from actont
template = cv.imread('T.png', 0)
w, h = template.shape[::-1]

# Apply template Matching
res = cv.matchTemplate(actont, template, cv.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

print(min_val, max_val, min_loc, max_loc)

cv.rectangle(actont, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,0), 2)

# Plot actont and res
plt.subplot(121), plt.imshow(actont, cmap='gray')
plt.title('actont'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.show()

# Cannot match the right position of letter 'T' from actont. See in the plot.