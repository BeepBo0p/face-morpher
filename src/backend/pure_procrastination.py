"""
Made a cool error when implementing the inverse distance weighting.
By interpolating colour values, but returning a single value using np.sum (which sums over the rgb-channels), I got a grayscale image.
The grayscale value did not give the right interpolation, but it made a really cool effect.
"""

from img_utils import *
import numpy as np
import copy
import os
import detect_face_features as dff


def inverse_distance_weighting(point: np.array, interpolants: np.ndarray, interpolants_value: np.ndarray, q: float) -> np.ndarray:
    """Given a point and a set of interpolants (and their associated value) and a q value, 
    computes the interpolated value at the given point. Increasing q will increase the influence of the closest points.

    Args:
        point (np.array): Point to compute the interpolated value at.
        interpolants (np.ndarray): Coordinates of the interpolants.
        interpolants_value (np.ndarray): Value of the interpolants.
        q (float): Parameter to control the influence of the closest points.

    Returns:
        np.ndarray: Interpolate value at the given point.
    """
    
    # First we handle the trivial case where the point is one of the interpolants
    for i in range(interpolants.shape[0]):
        if(np.array_equal(point, interpolants[i])):
            return interpolants_value[i]
        
    # Otherwise, actual interpolation must be done
    
    # Start by computing the distance between the point and each of the interpolants
    l2_distances = np.zeros((interpolants.shape[0]))
    
    for i in range(interpolants.shape[0]):
        l2_distances[i] = np.linalg.norm(point - interpolants[i])
        
    # Weights are the inverse of the distances raised to the power of q
    weights = 1 / (l2_distances ** q)
    
    # Normalise the weights
    weights = weights / np.sum(weights)
    
    # Obtain the weighted values through pairwise multiplication
    weighted_values = [weights[i] * interpolants_value[i] for i in range(interpolants.shape[0])]
    
        
    interpolated_value = np.sum(weighted_values)
    
    
        
    # Return the sum of the weighted values as the interpolated value
    return interpolated_value


def test_inverse_distance_weighting():
        
    print("=========== Testing inverse distance weighting ===========")
    
    # Test case 1
    point = np.array([0,0])
    interpolants = np.array([[0,0], [2,0], [0,2], [2,2]])
    interpolants_value = np.array([0, 1, 2, 3])
    q = 2
    
    #print(f'Interpolated value at {point}: {inverse_distance_weighting(point, interpolants, interpolants_value, q)}')
    
    # Test case 2
    point = np.array([1,1])
    interpolants = np.array([[0,0], [2,0], [0,2], [2,2]])
    interpolants_value = np.array([0, 1, 2, 3])
    q = 2
    
    #print(f'Interpolated value at {point}: {inverse_distance_weighting(point, interpolants, interpolants_value, q)}')
    
    # Test case 3
    point = np.array([0.5,0.5])
    interpolants = np.array([[0,0], [2,0], [0,2], [2,2]])
    interpolants_value = np.array([0, 1, 2, 3])
    q = 2
    
    #print(f'Interpolated value at {point}: {inverse_distance_weighting(point, interpolants, interpolants_value, q)}')
    
    # Image test case
    project_path = os.getcwd()
    output_path = os.path.join(project_path, 'output')
    
    dim = 1000
    dim1 = dim
    dim2 = dim
    
    img = Image(dim1, dim2, 3)
    
    num_interpolants = 100
    
    # Create random set of points and values within the image
    interpolants_x = np.random.randint(0, dim1, (num_interpolants, 1))
    interpolants_y = np.random.randint(0, dim2, (num_interpolants, 1))
    
    interpolants = np.concatenate((interpolants_x, interpolants_y), axis=1)
    interpolants_value = np.random.randint(0, 255, (num_interpolants, 3))
    
    # Set the pixels
    for i in range(interpolants.shape[0]):
        x, y = interpolants[i]
        img.data[x][y] = interpolants_value[i]
        
    # Save the image
    save_image(img, os.path.join(output_path, 'gen-test.png'))
    
    # Interpolate the rest of the image
    for x in range(img.width):
        for y in range(img.height):
            
            # Skip the pixels that are already interpolated
            if(img.data[x][y].any()):
                continue
            
            # Get the interpolated value
            value = inverse_distance_weighting(np.array([x,y]), interpolants, interpolants_value, 2)
            
            # Convert the value to a colour pixel
            value = np.array([value] * 3)
            
            # Scale according to rgb-channel
            R = 1
            G = 1
            B = 1
            
            value = value * np.array([R, G, B])
            
            img.data[x][y] = value
            
    # Filter the original interpolants
    for i in range(interpolants.shape[0]):
        x, y = interpolants[i]
        
        # Randomly set them as black or white
        if(np.random.randint(0,2)):
            img.data[x][y] = np.array([0,0,0])
        else:
            img.data[x][y] = np.array([255,255,255])

        
        
    # Normalize the image
    img.data = img.data - np.min(img.data)
    img.data = img.data / np.max(img.data)
            
    # Save the image
    save_image(img, os.path.join(output_path, 'gen.png'))


# Test the different functions
if __name__ == "__main__":
    
    #test_bilinear_sampling()
    #test_inverse_distance_interpolation()
    test_inverse_distance_weighting()