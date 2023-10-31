"""
This file contains the code to run the backend pipeline without the GUI.
It will be used for testing the project to ensure backend logic works as expected.
"""
import detect_face_features as dff
import interpolate as interp
import project_to_gan as ptg
import os
import cv2 as cv
import imageio as io
import numpy as np
import matplotlib.pyplot as plt


# Step 0. Define settings for pipeline run

# Image settings
target_resolution = (256, 256)
img1_name = "dc.jpg"
img2_name = "jc.jpg"

# Path variables
project_path = os.getcwd()
data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, 'output')
img1_path = os.path.join(data_path, img1_name)
img2_path = os.path.join(data_path, img2_name)

# Pipeline settings
interpolation_steps = 2
idw_q_parameter = 2
gan_refinement_steps = 10

# Video/GIF settings
output_name = "interpolation.gif"

def pre_process(
    img1_path: str,
    img2_path: str,
    dst_path: str,
    target_resolution: (int, int),
    ) -> tuple[np.ndarray, np.ndarray , list[np.ndarray], list[np.ndarray]]:

    # Step 1. Load the images, verify that they are valid images (dimensions > 0 and 3 channels. Also dim(A) == dim(B))))

    img1_path = os.path.join(data_path, img1_name)
    img2_path = os.path.join(data_path, img2_name)

    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    print('==================== Images loaded ====================')

    # Validate the images

    not_none = img1 is not None and img2 is not None
    not_empty = img1.size > 0 and img2.size > 0
    is_3_channel = len(img1.shape) == 3 and len(img2.shape) == 3
    is_same_size = img1.shape == img2.shape
    
    valid = not_none and not_empty and is_3_channel and is_same_size

    if(not valid):
        raise Exception("Invalid image(s).")
    
    # Resize the images to the target resolution using cv2
    img1 = cv.resize(img1, target_resolution)
    img2 = cv.resize(img2, target_resolution)

    # Show both images
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()

    img_list = [img1, img2]

    # Step 2. Detect the facial landmarks for both images.

    facial_features_list = [[] for i in range(len(img_list))]

    for i in range(len(img_list)):
        facial_features_list[i] = dff.detect_facial_features(img_list[i])

    print('==================== Facial features detected ====================')
    
    return img1, img2, facial_features_list[0], facial_features_list[1]

def morph_faces(
    img1: np.ndarray,
    img2: np.ndarray,
    facial_features_list: list[np.ndarray],
    output_path: str,
    output_name: str,
    interpolation_steps: int = interpolation_steps,
    idw_q_parameter: int = idw_q_parameter,
    gan_refinement_steps: int = gan_refinement_steps
    ) -> bool:

    # Step 3. Interpolate the facial landmarks using IDW.

    interpolation_path = os.path.join(output_path, 'interpolation')

    interpolation_path, interpolated_images = interp.inverse_distance_interpolation(
        img1,
        img2,
        facial_features_list[0],
        facial_features_list[1],
        n=interpolation_steps,
        q=idw_q_parameter,
        out_dir=interpolation_path
        )

    print('==================== Interpolation sequence generated ====================')

    # Step 4. Project the interpolated facial landmarks to the GAN.

    projection_sequence = []
    
    projection_path = os.path.join(output_path, 'projection')

    for image in interpolated_images:
        ptg.project_to_gan(
            src_path=interpolation_path,
            outdir=projection_path,
            save_video=False,
            seed=303,
            num_steps=gan_refinement_steps,
            output_name=image
        )
        
        projection_sequence.append(plt.imread(os.path.join(projection_path, image)))

    print('==================== Interpolation sequence refined with StyleGAN ====================')

    # Step 5. Save the interpolated images to a gif.

    io.mimsave(
        uri=os.path.join(output_path, output_name),
        ims=projection_sequence,
    )

    print('==================== Face morphing sequence saved to gif ====================')


if __name__ == '__main__':
    
    img1, img2, img1_features, img2_features = pre_process(
                                                            img1_path=img1_path,
                                                            img2_path=img2_path,
                                                            dst_path=output_path,
                                                            target_resolution=target_resolution
                                                            )
    
    # Adjust feature points here
    
    morph_faces(
                img1=img1,
                img2=img2,
                facial_features_list=[img1_features, img2_features],
                output_path=output_path,
                output_name=output_name,
                interpolation_steps=interpolation_steps,
                idw_q_parameter=idw_q_parameter,
                gan_refinement_steps=gan_refinement_steps
                )   