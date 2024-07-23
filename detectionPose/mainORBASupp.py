import cv2
import matplotlib.pyplot as plt
import numpy as np

import illumination

# Load the images
image1 = cv2.imread('./Dataset/id_pose_8_19_25_33/02001.png')
image2 = cv2.imread('./Dataset/id_pose_8_19_25_33/03005.png')
# image1 = cv2.imread('./Dataset/id_pose_4_8_12_16_20_24/DSC_2781.JPG')
# image2 = cv2.imread('./Dataset/id_pose_4_8_12_16_20_24/DSC_2782.JPG')


# Convert the training image to RGB
training_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
training_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Convert the training image to gray scale
training_gray1 = cv2.cvtColor(training_image1, cv2.COLOR_BGR2GRAY)
training_gray2 = cv2.cvtColor(training_image2, cv2.COLOR_BGR2GRAY)

training_gray1, training_gray2 = illumination.illumination_correction(training_gray1, training_gray2)

# # Display traning image and testing image
# fx, plots = plt.subplots(1, 2, figsize=(20,10))

# plots[0].set_title("Training Image")
# plots[0].imshow(training_image1)

# plots[1].set_title("Testing Image")
# plots[1].imshow(training_image2)


orb = cv2.ORB_create()
train_keypoints, train_descriptor = orb.detectAndCompute(training_gray1, None)
test_keypoints, test_descriptor = orb.detectAndCompute(training_gray2, None)

keypoints_without_size = np.copy(training_image1)
keypoints_with_size = np.copy(training_image1)

cv2.drawKeypoints(training_image1, train_keypoints, keypoints_without_size, color = (0, 255, 0))

cv2.drawKeypoints(training_image1, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Display image with and without keypoints size
# fx, plots = plt.subplots(1, 2, figsize=(20,10))

# plots[0].set_title("Train keypoints With Size")
# plots[0].imshow(keypoints_with_size, cmap='gray')

# plots[1].set_title("Train keypoints Without Size")
# plots[1].imshow(keypoints_without_size, cmap='gray')

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))

# Montrer les keypoints
img1 = cv2.drawKeypoints(training_image1, train_keypoints, None)
img2 = cv2.drawKeypoints(training_image2, test_keypoints, None)
plt.figure()
plt.imshow(img1)
plt.figure()
plt.imshow(img2)
plt.show()

# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the test image
matches = bf.match(train_descriptor, test_descriptor)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(training_image1, train_keypoints, training_image2, test_keypoints, matches, training_gray2, flags = 2)

# Display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))