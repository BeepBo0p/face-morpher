"""
This file contains methods needed to interpolate 2 images using IDW on the facial landmarks.
"""
from img_utils import *
import numpy as np
import copy

def inverse_distance_interpolation(img1: Image, img2: Image, features1: np.ndarray, features2: np.ndarray, alpha: float):
    pass


def bilinear_sampling(img: Image, x: float, y: float) -> np.ndarray:
    """Bilinearly samples a value from the image at the given coordinates.
    Assumes [0,0] is the top left corner of the image and img.width, img.height is the bottom right corner of the image.
    Padding is done by repeating the edge pixels and interpolation out of bounds is projected onto the edge.
    
    coordinate example with 1x1 image:
    
    (0,0)----------------------(W,0)
      |                          |
      |                          |
      |                          |
      |                          |
      |          (0,0)           |
      |                          |
      |                          |
      |                          |
      |                          |
    (0,H)----------------------(W,H)

    Args:
        img (Image): image to sample from
        x (float): position along width axis
        y (float): position along height axis

    Returns:
        np.ndarray: _description_
    """
    
    # First we deep copy the image and pad it
    img_padded = copy.deepcopy(img.data)
    
    img_padded = np.pad(img_padded, 1, mode='edge')
    
    
    # If the coordinates are out of bounds, project them onto the edge
    if(x < 0):
        x = 0
    elif(x > img.width):
        x = img.width - 1
        
    if(y < 0):
        y = 0
    elif(y > img.height):
        y = img.height - 1
        
    # Now we must find the 4 nearest pixels to the given coordinates
    # We start by transforming the coordinates from pixel edges to pixel centers
    
    w,h = img.width, img.height
    
    x_t, y_t = coordinate_transform(x, y, w+1, h+1, w, h)
    
    print(f'x_t: {x_t}, y_t: {y_t}')
    
    # if the coordinates are integers, we are done and can return the value at the given coordinates (accounting for padding)
    if x_t == int(x_t) and y_t == int(y_t):
        return img_padded[int(x_t)+1][int(y_t)+1]
        
    sample = np.array([x_t, y_t])
    
    top_left_x = np.array([np.floor(x_t), np.floor(y_t)])
    top_right_x = np.array([np.ceil(x_t), np.floor(y_t)])
    bottom_left_x = np.array([np.floor(x_t), np.ceil(y_t)])
    bottom_right_x = np.array([np.ceil(x_t), np.ceil(y_t)])
    
    nearest_pixels = np.array([top_left_x, top_right_x, bottom_left_x, bottom_right_x])
    
    for pixel in nearest_pixels:
        print(f'Nearest pixel: {pixel}', end=' | ')
    print("")
    
    # Cast the coordinates to integers
    nearest_pixels = nearest_pixels.astype(int)
    
    # Now we must find the distance from the given coordinates to each of the nearest pixels
    distances = np.zeros((4))
    
    for i in range(4):
        #compute euclidean distance and save to distance vector
        distances[i] = np.linalg.norm(nearest_pixels[i] - sample)
        
    nearest_values = []
    
    for pixel in nearest_pixels:
        
        i, j = pixel[0], pixel[1]
                
        # Since the image is padded we must add 1 to the coordinates
        nearest_values.append(img_padded[i][j])
        
    # we then normalise the distances
    distances = distances / np.sum(distances)
    
    
    # Define epsilon to within floating point error
    epsilon = 1e-10
    
    assert 1 - np.sum(distances) < epsilon
    
    # Initialise our sample value
    sampled_value = nearest_values[0] * 0    
    
    for i in range(4):
        
        weight = 1 - distances[i]
        
        sampled_value += nearest_values[i] * weight
        
    return sampled_value

    
def coordinate_transform(x: float, y: float, width_src: int, height_src: int, width_dst: int, height_dst: int) -> (float, float):
    """Computes coordinate transform from source image to destination image.
    Assumes that the source is bigger than the destination.

    Args:
        x (float): _description_
        y (float): _description_
        width_src (int): _description_
        height_src (int): _description_
        width_dst (int): _description_
        height_dst (int): _description_
    """
    # Find width and height difference
    width_diff = width_src - width_dst
    height_diff = height_src - height_dst
    
    # If width or height is within difference/2 of the edge, project onto the edge
    if(x < width_diff/2):
        x_t = 0
    elif(x > width_src - width_diff/2):
        x_t = width_src - 1
        
    if(y < height_diff/2):
        y_t = 0
    elif(y > height_src - height_diff/2):
        y_t = height_src - 1
        
    # Otherwise, find how far along the image the coordinates are and scale them to the source image
    x_t = ((x + 1) / width_dst) * width_src - 1
    y_t = ((y + 1) / height_dst) * height_src - 1
    
    return x_t, y_t
    
    pass
    
    
# Test the different functions
if __name__ == "__main__":
    
    print("=========== Testing bilinear sampling ===========")
    
    sample_image = Image(2,2,1)
    
    # Set the pixels
    sample_image.data = np.array([[0,1], [2,3]])
    
    """     for row in sample_image.data:
            print(row) """    
            
    outer_corners = np.array([[0,0], [1,0], [0,1], [1,1]])
    pixel_centers = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    
    for i in outer_corners:
        print(f'Sampled value at {i[0]}, {i[1]}: {bilinear_sampling(sample_image, i[0], i[1])}')
        print('\n')
        
    """     for i in pixel_centers:
            print(f'Sampled value at {i[0]}, {i[1]}: {bilinear_sampling(sample_image, i[0], i[1])}')
    """                
    
    mean = np.mean(sample_image.data)
    
    mean_sample = bilinear_sampling(sample_image, 0.5, 0.5)
    
    print(f'Mean: {mean}, Mean sample: {mean_sample}')