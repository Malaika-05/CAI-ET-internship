# Import Libraries

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Task 1: Read an Image
image = cv2.imread("images\sample.png")

# What happens?
# OpenCV reads the image as a NumPy array.
# Every image is just numbers.
# (OpenCV uses BGR instead of RGB.)


# Task 2 Display Image
cv2.imshow("Original Image", image) #Displays image in a window.

cv2.waitKey(0) #Waits until a key is pressed.

cv2.destroyAllWindows()#Closes window.


# Task 3 Save Image
# cv2.imwrite("output/original.jpg", image) #Writes image to disk.

# Task 4 Resize Image
resized = cv2.resize(image, (300,100))
cv2.imwrite("output/resized.jpg", resized) #Writes resized image to disk.
# Useful before training AI models because neural networks require fixed image sizes.

# Crop Image
cropped = image[100:400, 100:500]
cv2.imwrite("output/cropped.jpg", cropped)
# Image is a NumPy matrix.
# Rows = height
# Columns = width

# Rotate Image
height, width = image.shape[:2]

center = (width//2, height//2)

matrix = cv2.getRotationMatrix2D(center,45,1) #rotate the image by 45 degrees with a scale of 1

rotated = cv2.warpAffine(image,matrix,(width,height))

cv2.imwrite("output/rotated.jpg",rotated)

# Convert to Grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imwrite("output/grayscale.jpg",gray)

# Task 8 Blur Image
blur = cv2.GaussianBlur(gray,(7,7),0)

cv2.imwrite("output/blur.jpg",blur)

#Edge Detection
edges = cv2.Canny(blur,100,200)

cv2.imwrite("output/edges.jpg",edges)

# Display Using Matplotlib
plt.imshow(edges,cmap="gray")

plt.title("Edge Detection")

plt.axis("off")

plt.show()