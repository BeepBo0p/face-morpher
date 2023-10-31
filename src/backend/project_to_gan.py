"""
This file contains the code for projecting the interpolated images to a pre-trained GAN.

The procedure will do the following:

1. Load the pre-trained GAN.

2. Load an interpolated image.

3. Initalize a random latent vector.

4. Measure the loss between the interpolated image and the generated image from the latent vector.

5. Optimize the latent vector to minimize the loss.

6. Repeat steps 4 and 5 until the loss is below a certain threshold.

7. Save the image generated from the optimized latent vector.

Luckily, NVIDIA has already made this for us (seemingly based on the image2stylegan paper), so we just call their code.

"""
import backend.projector as projector
import os

def project_to_gan(src_path: str='output/interpolation', 
                   outdir: str='output/projection', 
                   save_video: bool=False, 
                   seed: int=303, 
                   num_steps: int=500, 
                   output_name: str='interpolated_image.png'):

    src_path = os.path.abspath('output/interpolation')
    
    print(f'src_path: {src_path}')

    interpolated_images = [i for i in os.listdir(src_path) if i.endswith('.png')]
    interpolated_images.sort()

    for image in interpolated_images:
        
        print(f'Projecting {image} to GAN...')
        
        projector.run_projection(
            network_pkl='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
            target_fname=os.path.join(src_path, image),
            outdir ='output/projection',
            save_video=False,
            seed=303,
            num_steps=500,
            output_name=image
        )
    
if __name__ == '__main__':
    project_to_gan()