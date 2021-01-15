import cv2
import numpy as np

filename = './data/chessboard_small.png'


img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray32 = np.float32(gray)


####### Harris Corner detection
#   cv2.cornerHarris(img, blockSize, ksize, k)
#    
#    img - Input image, it should be grayscale and float32 type.
#    blockSize - It is the size of neighbourhood considered for corner detection
#    ksize - Aperture parameter of Sobel derivative used.
#    k - Harris detector free parameter in the equation.

dst = cv2.cornerHarris(gray32,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)  # OpenCV 2.4.13
dst = cv2.dilate(dst)  # OpenCV 3

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.namedWindow( 'dst', cv2.WINDOW_AUTOSIZE);
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# 1. Display the Harris point on other images (lena, box,...)


####### MSER region detection
### OpenCV 2.4.13
mser = cv2.MSER()
regions = mser.detect(gray, None)

#### OpenCV 3
#mser = cv2.MSER_create()
#regions = mser.detectRegions(gray, None)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(gray, hulls, 1, (255, 0, 0))

cv2.imshow('MSER', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()