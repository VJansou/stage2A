import cv2
import numpy as np
import matplotlib.pyplot as plt

from consts import *
import utils
import illumination

# Load the images
image1 = cv2.imread('./Dataset/id_pose_5_10_15_20/02001.png')
image2 = cv2.imread('./Dataset/id_pose_5_10_15_20/03005.png')

# Convert the training image to gray scale
training_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
training_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

training_gray1, training_gray2 = illumination.illumination_correction(training_gray1, training_gray2)

# Detect keypoints and create descriptors
sift = cv2.SIFT.create()

train_keypoints, train_descriptor = sift.detectAndCompute(training_gray1, None)
test_keypoints, test_descriptor = sift.detectAndCompute(training_gray2, None)

# Display keypoints
img1 = cv2.drawKeypoints(training_gray1, train_keypoints, training_gray1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints(training_gray2, test_keypoints, training_gray2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure()
plt.imshow(img1, cmap='gray')
plt.figure()
plt.imshow(img2, cmap='gray')
plt.show()

# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the test image
matches = bf.match(train_descriptor, test_descriptor)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), train_keypoints, cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), test_keypoints, matches, training_gray2, flags = 2)

# Display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))