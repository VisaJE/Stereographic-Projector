import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt

img = cv2.imread('path/to/image.png',cv2.IMREAD_GRAYSCALE)
imgc = img.copy()

# If edges are used, one has to remove duplicate contours
edges = cv2.Canny(img,100,200)

cv2.imshow('Canny before Contouring', edges)
cv2.waitKey(0)


kernel = np.ones((3,3),np.float32)/9
dst = cv2.filter2D(edges,-1,kernel)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(dst,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
for i in range(len(contours)):
    contours[i] = cv2.approxPolyDP(contours[i], 4, True)

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(imgc, contours, -1, (100, 255, 0), 2)

cv2.imshow('Contours', imgc)
cv2.waitKey(0)
with open("contours.dat", 'wb+') as f:
    pickle.dump(contours,f)
