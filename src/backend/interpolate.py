"""
This file contains methods needed to interpolate 2 images using IDW on the facial landmarks.
"""
from backend.img_utils import *
import numpy as np
import copy

def get_delta(x: np.array, y: np.array) -> np.array:
    """Computes the delta vector between the 2 given vectors.
    The delta vector is the difference between the 2 vectors.
    
    Args:
        x (np.array): _description_
        y (np.array): _description_
    
    Returns:
        np.array: _description_
    """
    return y - x


def inverse_distance_interpolation(img1: Image, img2: Image, features1: np.ndarray, features2: np.ndarray, n: int, q: float) -> list[Image]:
    
    # Make sure the features map 1 to 1
    assert features1.shape == features2.shape
    
    # Get the delta vector between the 2 feature maps
    feature_delta = get_delta(features1, features2)
    
    # Create the steps for the interpolation
    timesteps = np.linspace(0, 1, n)
    
    print(f'Timesteps: {timesteps}')
    
    # List to store the interpolated images
    slices = []
    
    for t in timesteps:
        
        # Compute the interpolated features for the current timestep
        interpolated_features = features1 + t * feature_delta

        # Define the delta field for the image
        delta_field = np.zeros((img1.width, img1.height, 2))
        
        # TODO: Verify that this is the right approach, I don't think it is
        # Insert each interpolated feature into the delta field at its original position
        for i in range(features1.shape[0]):
            
            x, y = features1[i]
            
            delta_field[x][y] = interpolated_features[i]
            
            
        # Interpolate the delta field using IDW
        delta_field = inverse_distance_weighting(img1, img2, delta_field, q)
        
        # sample from both images and use the delta field to linearly interpolate between them
        
        interpolated_image = np.zeros((img1.width, img1.height))
        
        

                
                
                
        
        
                
        
    
    
def inverse_distance_weighting(img1: Image, img2: Image, delta_field: np.ndarray, q: float) -> np.ndarray:
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
    
    # First we deep copy the image and pad it. This should allow us to handle the edge cases more easily.
    # NOTE: When actually sampling we must add 1 to the coordinates to account for the padding.
    img_padded = copy.deepcopy(img.data)
    img_padded = np.pad(img_padded, 1, mode='edge')
    
    
    # If the coordinates are out of bounds, project them onto the nearest edge
    if(x < 0):
        x = 0
    elif(x > img.width):
        x = img.width - 1
        
    if(y < 0):
        y = 0
    elif(y > img.height):
        y = img.height - 1
        
    # Now we must find the 4 nearest pixels to the given coordinates

    top_edge = round(y - 0.00001) == 0
    bottom_edge = round(y + 0.00001) == img.height
    left_edge = round(x - 0.00001) == 0
    right_edge = round(x + 0.00001) == img.width
    
    top_left = top_edge and left_edge
    top_right = top_edge and right_edge
    bottom_left = bottom_edge and left_edge
    bottom_right = bottom_edge and right_edge
    
    if(top_left):
        #print("Top left corner")
        return img_padded[1][1]
    
    if(top_right):
        #print("Top right corner")
        return img_padded[-2][1]
    
    if(bottom_left):
        #print("Bottom left corner")
        return img_padded[1][-2]
    
    if(bottom_right):
        #print("Bottom right corner")
        return img_padded[-2][-2]
    
    # If we reach this point, we know that the coordinates are not on the corners of the image
    
    if(top_edge):
        #print("Top edge")
        y = 1
        x_s = x - 0.5
        x_s += 1 # add 1 to account for padding
        
        left_x = np.floor(x_s).astype(int)
        right_x = np.ceil(x_s).astype(int)
        
        left_pixel = img_padded[left_x][y]
        right_pixel = img_padded[right_x][y]
        
        left_dist = np.abs(left_x - x_s)
        right_dist = np.abs(right_x - x_s)
        
        sum_dist = left_dist + right_dist
        
        if(sum_dist == 0):
            return left_pixel
        
        left_weight = 1 - (left_dist / sum_dist)
        right_weight = 1 - (right_dist / sum_dist)
        
        return left_pixel * left_weight + right_pixel * right_weight
    
    if(bottom_edge):
        #print("Bottom edge")
        y = -2
        x_s = x - 0.5
        x_s += 1 # add 1 to account for padding
        
        left_x = np.floor(x_s).astype(int)
        right_x = np.ceil(x_s).astype(int)
        
        left_pixel = img_padded[left_x][y]
        right_pixel = img_padded[right_x][y]
        
        left_dist = np.abs(left_x - x_s)
        right_dist = np.abs(right_x - x_s)
        
        sum_dist = left_dist + right_dist
        
        if(sum_dist == 0):
            return left_pixel
        
        left_weight = 1 - (left_dist / sum_dist)
        right_weight = 1 - (right_dist / sum_dist)
        
        return left_pixel * left_weight + right_pixel * right_weight
    
    if(left_edge):
        #print("Left edge")
        x = 1
        y_s = y - 0.5
        y_s += 1 # add 1 to account for padding
        
        top_y = np.floor(y_s).astype(int)
        bottom_y = np.ceil(y_s).astype(int)

        
        top_pixel = img_padded[x][top_y]
        bottom_pixel = img_padded[x][bottom_y]
        
        top_dist = np.abs(top_y - y_s)
        bottom_dist = np.abs(bottom_y - y_s)
        
        sum_dist = top_dist + bottom_dist
        
        if(sum_dist == 0):
            return top_pixel
        
        top_weight = 1 - (top_dist / sum_dist)
        bottom_weight = 1 - (bottom_dist / sum_dist)
        
        return top_pixel * top_weight + bottom_pixel * bottom_weight
    
    if(right_edge):
        #print("Right edge")
        x = -2
        y_s = y - 0.5
        y_s += 1 # add 1 to account for padding
        
        top_y = np.floor(y_s).astype(int)
        bottom_y = np.ceil(y_s).astype(int)
        
        top_pixel = img_padded[x][top_y]
        bottom_pixel = img_padded[x][bottom_y]
        
        top_dist = np.abs(top_y - y_s)
        bottom_dist = np.abs(bottom_y - y_s)
        
        sum_dist = top_dist + bottom_dist
        
        if(sum_dist == 0):
            return top_pixel
        
        top_weight = 1 - (top_dist / sum_dist)
        bottom_weight = 1 - (bottom_dist / sum_dist)
        
        return top_pixel * top_weight + bottom_pixel * bottom_weight

        
        
    # If we reach this point, we know that the coordinates are not on the edges of the image
    # Now we can safely transform the coordinates to the original image
    
    x_t = x - 0.5
    y_t = y - 0.5
    
    # We padded the image so we must add 1 to the coordinates
    x_t += 1
    y_t += 1
    
    sample = np.array([x_t, y_t])
    
    top_left_px = np.array([np.floor(x_t), np.floor(y_t)])
    top_right_px = np.array([np.ceil(x_t), np.floor(y_t)])
    bottom_left_px = np.array([np.floor(x_t), np.ceil(y_t)])
    bottom_right_px = np.array([np.ceil(x_t), np.ceil(y_t)])
    
    nearest_pixels = np.array([top_left_px, top_right_px, bottom_left_px, bottom_right_px])
    
    #for pixel in nearest_pixels:
    #    print(f'Nearest pixel: {pixel}', end=' | ')
    #print("")
    
    # Cast the coordinates to integers
    nearest_pixels = nearest_pixels.astype(int)
    
    # Now we must find the distance from the given coordinates to each of the nearest pixels
    distances = np.zeros((4))
    
    for i in range(4):
        #compute euclidean distance and save to distance vector
        distances[i] = np.linalg.norm(nearest_pixels[i] - sample)   
        #print(f'Calculated distance: {np.linalg.norm(nearest_pixels[i] - sample)}', end=' | ')
    #print("")
        
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
    
    # we solve a matrix equation to find the weights for each pixel
    
    vec = np.array([1, x_t, y_t, x_t * y_t])
    
    x_1 = nearest_pixels[0][0]
    y_1 = nearest_pixels[0][1]
    x_2 = nearest_pixels[-1][0]
    y_2 = nearest_pixels[-1][1]
    

    
    weights = np.array([
        ((x_2 - x_t) * (y_2 - y_t))/((x_2 - x_1) * (y_2 - y_1)),
        ((x_t - x_1) * (y_2 - y_t))/((x_2 - x_1) * (y_2 - y_1)),
        ((x_2 - x_t) * (y_t - y_1))/((x_2 - x_1) * (y_2 - y_1)),
        ((x_t - x_1) * (y_t - y_1))/((x_2 - x_1) * (y_2 - y_1))
    ])
    
    for i in range(4):
        
        weight = weights[i]
        
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


def test_inverse_distance_interpolation():
    
    print("=========== Testing inverse distance interpolation ===========")
    
    img1 = Image(2,2,1)
    img2 = Image(2,2,1)
    
    # Set the pixels
    img1.data = np.array([[0,1], [2,3]])
    img2.data = np.array([[4,5], [6,7]])
    
    # Set the features
    features1 = np.array([[0,0], [1,0], [0,1], [1,1]])
    features2 = np.array([[0,0], [1,0], [0,1], [1,1]])
    
    n = 10
    
    interpolated_image = inverse_distance_interpolation(img1, img2, features1, features2, n, q=0.5)
    
    #plt.imshow(interpolated_image, cmap='gray')
    #plt.show()    
    return


def test_bilinear_sampling():
    
    print("=========== Testing bilinear sampling ===========")
    
    sample_image = Image(2,2,1)
    
    # Set the pixels
    sample_image.data = np.array([[0,1], [2,3]])
    
    """     for row in sample_image.data:
            print(row) """    
            
    outer_corners = np.array([[0,0], [1,0], [0,1], [1,1]])
    pixel_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    
    for i in outer_corners:
        print(f'Sampled value at {i[0]}, {i[1]}: {bilinear_sampling(sample_image, i[0], i[1])}')
        print('\n')
        
    for i in pixel_centers:
            print(f'Sampled value at {i[0]}, {i[1]}: {bilinear_sampling(sample_image, i[0], i[1])}')
                    
    
    mean = np.mean(sample_image.data)
    
    mean_sample = bilinear_sampling(sample_image, 1, 1)
    
    print(f'Mean: {mean}, Mean sample: {mean_sample}')
    
    almost_mean = bilinear_sampling(sample_image, 1.1, 1)
    
    print(f'Almost mean: {almost_mean}')
    
    resolution = 500

    
    sampled_image = np.zeros((resolution+1, resolution+1))
    
    for h in range(resolution + 1):
        h_t = h / resolution * 2
        
        for w in range(resolution + 1):
            w_t = w / resolution * 2
            
            #print(f'Sampling at {w_t}, {h_t}')
            sampled_image[w][h] = bilinear_sampling(sample_image, w_t, h_t)        
        
        
    plt.imshow(sampled_image, cmap='gray')
    plt.show()
    
# Test the different functions
if __name__ == "__main__":
    
    #test_bilinear_sampling()
    test_inverse_distance_interpolation()