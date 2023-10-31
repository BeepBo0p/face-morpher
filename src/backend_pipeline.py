"""
This file contains the code to run the backend pipeline without the GUI.
It will be used for testing the project to ensure backend logic works as expected.
"""
from backend.img_utils import *
import backend.detect_face_features as dff
import backend.interpolate as interp
import backend.project_to_gan as ptg
import os
import imageio as io


# Step 0. Define settings for pipeline run

# Image settings
target_resolution = (256, 256)
img1_name = "dorde.jpg"
img2_name = "jørgen.jpg"

# Path variables
project_path = os.getcwd()
data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, 'output')
img1_path = os.path.join(data_path, img1_name)
img2_path = os.path.join(data_path, img2_name)

# Pipeline settings
interpolation_steps = 10
idw_q_parameter = 2
gan_refinement_steps = 500

# Video/GIF settings
output_name = "interpolation.gif"

def pre_process(
    img1_path: str,
    img2_path: str,
    dst_path: str,
    target_resolution: (int, int),
    ) -> (Image, Image , list(np.ndarray), list(np.ndarray)):

    # Step 1. Load the images, verify that they are valid images (dimensions > 0 and 3 channels. Also dim(A) == dim(B))))

    img1_path = os.path.join(data_path, img1_name)
    img2_path = os.path.join(data_path, img2_name)

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    print('==================== Images loaded ====================')

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

    facial_features_list = [[] for i in range(len(img_list))]

    for i in range(len(img_list)):
        facial_features_list[i] = dff.detect_facial_features(img_list[i])

    print('==================== Facial features detected ====================')
    
    return img1, img2, facial_features_list[0], facial_features_list[1]

def morph_faces(
    img1: Image,
    img2: Image,
    facial_features_list: list(np.ndarray),
    output_path: str,
    output_name: str,
    interpolation_steps: int = 10,
    idw_q_parameter: int = 2,
    gan_refinement_steps: int = 500
    ) -> bool:

    # Step 3. Interpolate the facial landmarks using IDW.

    interpolation_path = os.path.join(project_path, 'output/interpolation')

    interpolation_path, interpolated_images = interp.inverse_distance_interpolation(
        img1,
        img2,
        facial_features_list[0],
        facial_features_list[1],
        n=10,
        q=2,
        out_dir=interpolation_path
        )

    print('==================== Interpolation sequence generated ====================')

    # Step 4. Project the interpolated facial landmarks to the GAN.

    projection_sequence = []

    for image in interpolated_images:
        ptg.project_to_gan(
            src_path=interpolation_path,
            outdir=output_path,
            save_video=False,
            seed=303,
            num_steps=500,
            output_name=image
        )
        
        projection_sequence.append(plt.imread(os.path.join(output_path, image)))

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