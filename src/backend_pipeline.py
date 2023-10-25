"""
This file contains the code to run the backend pipeline without the GUI.
It will be used for testing the project to ensure backend logic works as expected.
"""
from backend.img_utils import *
import backend.detect_face_features as dff
import backend.interpolate as interp
import backend.project_to_gan as ptg


# Step 1. Load the images, verify that they are valid images (dimensions > 0 and 3 channels. Also dim(A) == dim(B))))
img1_path = "../data/test.jpg"
img2_path = "../data/test.jpg"

img1 = load_image(img1_path)
img2 = load_image(img2_path)

# Validate the images

if(not is_valid_image(img1) or not is_valid_image(img2)):
    raise Exception("Invalid image(s).")

if(not is_same_size(img1, img2)):
    raise Exception("Images must be the same size.")

# Show both images
show_image(img1)
show_image(img2)

img_list = [img1, img2]


# Step 2. Detect the facial landmarks for both images.

for img in img_list:
    facial_features = dff.detect_facial_features(img)
    print(facial_features)


# Step 3. Interpolate the facial landmarks using IDW.

# Step 4. Project the interpolated facial landmarks to the GAN.

