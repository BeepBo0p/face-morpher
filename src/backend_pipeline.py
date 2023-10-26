"""
This file contains the code to run the backend pipeline without the GUI.
It will be used for testing the project to ensure backend logic works as expected.
"""
from backend.img_utils import *
import backend.detect_face_features as dff
import backend.interpolate as interp
import backend.project_to_gan as ptg
import os


# Step 1. Load the images, verify that they are valid images (dimensions > 0 and 3 channels. Also dim(A) == dim(B))))
project_path = os.getcwd()
data_path = os.path.join(project_path, "data")

dorde_path = os.path.join(data_path, "dorde.jpg")
jørgen_path = os.path.join(data_path, "jørgen.jpg")


dorde = load_image(dorde_path)
jørgen = load_image(jørgen_path)

img1 = dorde
img2 = jørgen

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

