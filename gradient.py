#export PYTHONPATH="/usr/local/lib/python2.7/site-packages"

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an image in greyscale
img = cv2.imread('./data/rectangle.png',0)

# Compute gradient in X and Y directions
gradx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
grady=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

# Compute absolute values
abs_gradx = np.uint8(np.absolute(gradx))
abs_grady = np.uint8(np.absolute(grady))

plt.subplot(1,3,1),plt.imshow(img,cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(gradx,cmap='gray')
plt.title('Grad X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(grady,cmap='gray')
plt.title('Grad Y'), plt.xticks([]), plt.yticks([])
plt.show() # enlarge the figure

plt.subplot(1,3,1),plt.imshow(img,cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(abs_gradx,cmap='gray')
plt.title('abs(Grad X)'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(abs_grady,cmap='gray')
plt.title('abs(Grad Y)'), plt.xticks([]), plt.yticks([])
plt.show() # enlarge the figure


# Compute magnitude and angle of the gradient ( in degrees )
mag, angle=cv2.cartToPolar(gradx,grady,angleInDegrees=True)

plt.subplot(1,3,1),plt.imshow(img,cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(mag,cmap='gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(angle,cmap='gray')
plt.title('Orientation'), plt.xticks([]), plt.yticks([])
plt.show()


# Compute orientation histograms
hog = cv2.calcHist([img],[0],None,[360],[0,360])

plt.plot(hog)
plt.title('Orientation Histogram')
plt.show()

# 1. Add a mask to take into account only non null gradients
# 2. Quantize the histogram into 8 bins


##### Rotation 45 degrees
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
rotim = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('rotation',rotim)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3. Compute the quantized orientation histogram of the rotated image into 8 bins
# 4. Compare it with the unrotated version
