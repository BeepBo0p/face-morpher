"""
This file contains methods needed to interpolate 2 images using IDW on the facial landmarks.
"""

import numpy as np
import copy
import os
import backend.detect_face_features as dff
from numba import njit, jit
import cv2 as cv
import matplotlib.pyplot as plt

@njit
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


def inverse_distance_interpolation(img1: np.ndarray, img2: np.ndarray, features1: np.ndarray, features2: np.ndarray, n: int, q: float, out_dir: str) -> tuple[str, list[str]]:
    """Creates a sequence of interpolated images using facial features as guide.
    Features are interpolated are linearly interpolated between the 2 images along with the colour values at each time step t (out of n+1 steps).
    Each shifted feature is then used to compute a delta field using IDW to describe the shift of all other pixels in the image.
    The delta field describes the shift of each pixel in the image from the first image to the second image.
    Afterwards, the delta field is used to bilinearly sample colour from both images and linearly interpolate between them.
    The interpolated image is then stored in a list and the list is returned.
    

    Args:
        img1 (np.ndarray): image of the first person
        img2 (np.ndarray): image of the second person
        features1 (np.ndarray): detected facial features of the first person
        features2 (np.ndarray): detected facial features of the second person
        n (int): number of timesteps to interpolate between the 2 images
        q (float): parameter to control the influence of the closest points in the IDW interpolation

    Returns:
        tuple[str, list[str]]: tuple with the path to the output directory and a list of the filenames of the interpolated images
    """
    
    # Make sure the features map 1 to 1
    assert features1.shape == features2.shape
    
    
    # Normalise both images
    #max_val = np.max(img1)
    #min_val = np.min(img1)
    
    #img1 = (img1 - min_val) / (max_val - min_val)
    
    #max_val = np.max(img2)
    #min_val = np.min(img2)
    
    #img2 = (img2 - min_val) / (max_val - min_val)
    
    # Get the positional delta vector between the 2 feature maps
    feature_pos_delta = get_delta(features1, features2)
    
    # Get the colour values for each feature set
    img1_colours = np.array([img1[x][y] for x, y in features1])
    img2_colours = np.array([img2[x][y] for x, y in features2])
    
    # Get the colour delta vector between the 2 images
    feature_col_delta = get_delta(img1_colours, img2_colours)
    
    
    # Create the steps for the interpolation
    timesteps = np.linspace(0, 1, n)
    
    print(f'Timesteps: {timesteps}')
    
    # List to store the interpolated images
    slices = []
    
    feature_list = [features1]
    colour_list = [img1_colours]
    
    interpolated_image_list = [img1]
    interpolated_delta_field_list = []
    
    image_nr = 0
    image_path_list = []
    
    for t in timesteps:
        
        w,h,c = img1.shape
        
        interpolated_image = np.zeros((w,h,c))
        
        # The delta field tracks how far the pixels have moved from the first image to the second image and from the second image to the first image
        interpolated_delta_field = np.zeros((w, h, 4))
        
        # Compute the interpolated feautres from image 1 in positive t-direction
        interpolated_pos_delta = feature_pos_delta * t
        
        # Compute the interpolated features from image 2 in negative t-direction
        reverse_interpolated_pos_delta = feature_pos_delta * (1 - t)
        
        #interpolated_col_delta = feature_col_delta * t
        
        interpolated_features = features1 + interpolated_pos_delta
        #interpolated_colours = img1_colours + interpolated_col_delta
        
        # Round features and colours to integers
        interpolated_features = interpolated_features.astype(int)
        #interpolated_colours = interpolated_colours
        
        feature_list.append(interpolated_features)
        #colour_list.append(interpolated_colours)
        
        # Print the first 5 interpolated features
        print(f'Interpolated features: {interpolated_features[:1]}', end=' | ')
        #print(f'Interpolated colours: {interpolated_colours[:1]}', end=' | ')
        
        # Store the deltas in positive and negative t-direction together
        interpolated_pos_delta = np.concatenate((-1 * interpolated_pos_delta, reverse_interpolated_pos_delta), axis=1)
        print(f'Interpolated delta: {interpolated_pos_delta[:1]}')
        
        for i in range(interpolated_features.shape[0]):
            
            coord = interpolated_features[i]
            
            x, y = coord
            
            # Store the values in the interpolated image
            #interpolated_image[x][y] = interpolated_colours[i]
            
            # Store the delta values to use for sampling later
            interpolated_delta_field[x][y] = interpolated_pos_delta[i]
            #TODO: Carefully check the sign of the delta field.            
            #TODO: Verify that we can use the same delta field for both images.
            
        # Next steps:
        # 1. Compute the rest of the delta field using IDW
        
        #print(f"Computing delta field for t={t}:", end=' ')
        
        for x in range(w):
            for y in range(h):
                
                # Skip the pixels that are already interpolated
                if(interpolated_image[x][y].any()):
                    continue
                                
                # Compute the delta field
                interpolated_delta_field[x][y] = inverse_distance_weighting(np.array([x,y], dtype=float), interpolated_features.astype(float), interpolated_pos_delta, q)
                #print('|', end=' ')
        
        print(f"Delta field computed for t={t}")
        
        # 2. Using the delta field, bilinearly sample from both images and linearly interpolate between them
        
        for x in range(w):
            for y in range(h):
                
                # Skip the pixels that are already interpolated
                if(interpolated_image[x][y].any()):
                    continue
                
                # Get the delta field at the current pixel
                delta = interpolated_delta_field[x][y]
                
                # Sample from the 2 images
                img1_sample = bilinear_sampling(img1.astype(float), float(x) + delta[0], float(y) + delta[1])
                
                #TODO: Verify that we can use the same delta field for both images
                img2_sample = bilinear_sampling(img2.astype(float), float(x) + delta[-2], float(y) + delta[-1])
                
                # weights are the time distance from the first to second image
                
                
                # Compute the bilinear sampling
                interpolated_image[x][y] = img1_sample * (1 - t) + img2_sample * t
        
        print(f"Interpolated image computed for t={t}")
        
        # 3. Store the interpolated image in a list
        
        #interpolated_image_list.append(interpolated_image)
        
        # Clear the figure
        #plt.clf()
        
        # Plot delta field as a heatmap
        #plt.imshow(interpolated_delta_field[:,:,0], cmap='hot', interpolation='nearest')
        
        # Save plot to file in high resolution
        #plt.savefig(os.path.join(os.getcwd(), f'output/interpolation/delta_field_{int(image_nr)}.png'), dpi=300)
        
        # Save the image to file
        cv.imwrite(os.path.join(os.getcwd(), f'output/interpolation/interpolated_image_{int(image_nr)}.png'), interpolated_image)
        image_path_list.append(f'interpolated_image_{int(image_nr)}.png')
        
        image_nr += 1
        
        
        
    # Append the second image to the list
    #interpolated_image_list.append(img2.data)
    cv.imwrite(os.path.join(os.getcwd(), f'output/interpolation/interpolated_image_{n}.png'), img2)
    image_path_list.append(f'interpolated_image_{n}.png')
    
    return (os.path.join(os.getcwd(), out_dir), image_path_list)
        

@njit#(parallel=True, fastmath=True, cache=True)
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
    
    interpolated_value = interpolants_value[0] * 0
    
    for i in range(interpolants.shape[0]):
        interpolated_value += weighted_values[i]
    
    # Return the sum of the weighted values as the interpolated value
    return interpolated_value
   
    
@njit#(parallel=True, fastmath=True, cache=True)
def bilinear_sampling(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinearly samples a value from the image at the given coordinates.
    Assumes [0,0] is the top left corner of the image and width, height is the bottom right corner of the image.
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
    img_padded = img
    width, height = img.shape[0], img.shape[1]
    
    
    # If the coordinates are out of bounds, project them onto the nearest edge
    if(x < 0):
        x = 0
    elif(x > width):
        x = width
        
    if(y < 0):
        y = 0
    elif(y > height):
        y = height
        
    # Now we must find the 4 nearest pixels to the given coordinates

    top_edge = round(y - 0.00001) == 0
    bottom_edge = round(y + 0.00001) == height
    left_edge = round(x - 0.00001) == 0
    right_edge = round(x + 0.00001) == width
    
    top_left = top_edge and left_edge
    top_right = top_edge and right_edge
    bottom_left = bottom_edge and left_edge
    bottom_right = bottom_edge and right_edge
    
    if(top_left):
        #print("Top left corner")
        return img_padded[0][0]
    
    if(top_right):
        #print("Top right corner")
        return img_padded[-1][0]
    
    if(bottom_left):
        #print("Bottom left corner")
        return img_padded[0][-1]
    
    if(bottom_right):
        #print("Bottom right corner")
        return img_padded[-1][-1]
    
    # If we reach this point, we know that the coordinates are not on the corners of the image
    
    if(top_edge):
        #print("Top edge")
        y = 0
        x_s = x - 0.5
        #x_s += 1 # add 1 to account for padding
        
        left_x = int(np.floor(x_s))
        right_x = int(np.ceil(x_s))
        
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
        y = -1
        x_s = x - 0.5
        #x_s += 1 # add 1 to account for padding
        
        left_x = int(np.floor(x_s))
        right_x = int(np.ceil(x_s))
        
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
        x = 0
        y_s = y - 0.5
        #y_s += 1 # add 1 to account for padding
        
        top_y = int(np.floor(y_s))
        bottom_y = int(np.ceil(y_s))

        
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
        x = -1
        y_s = y - 0.5
        #y_s += 1 # add 1 to account for padding
        
        top_y = int(np.floor(y_s))
        bottom_y = int(np.ceil(y_s))
        
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
    #x_t += 1
    #y_t += 1
    
    left_x = int(np.floor(x_t))
    right_x = int(np.ceil(x_t))
    top_y = int(np.floor(y_t))
    bottom_y = int(np.ceil(y_t))
    
    sample = np.array([x_t, y_t])
    
    top_left_px = np.array([left_x, top_y])
    top_right_px = np.array([right_x, top_y])
    bottom_left_px = np.array([left_x, bottom_y])
    bottom_right_px = np.array([right_x, bottom_y])
    
    nearest_pixels = [top_left_px, top_right_px, bottom_left_px, bottom_right_px]
    
    #for pixel in nearest_pixels:
    #    print(f'Nearest pixel: {pixel}', end=' | ')
    #print("")
    
    
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
        
        value = img_padded[i][j]
                
        nearest_values.append(value)
               
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
    
    project_path = os.getcwd()
    data_path = os.path.join(project_path, 'data')
    output_path = os.path.join(project_path, 'output/interpolation')
    
    dorde_path = os.path.join(data_path, 'dc.jpg')
    jørgen_path = os.path.join(data_path, 'jc.jpg')
    
    img1 = cv.imread(dorde_path)
    img2 = cv.imread(jørgen_path)
        
    # Resize the images to 256x256
    #img1.data = cv.resize(img1.data, (256, 256))
    #img2.data = cv.resize(img2.data, (256, 256))
    
    # Convert to standard RGB
    #img1.data = cv.cvtColor(img1.data, cv.COLOR_BGR2RGB)
    #img2.data = cv.cvtColor(img2.data, cv.COLOR_BGR2RGB)
    
    # Resize the images to 200x250
    #img1.data = cv.resize(img1.data, (200, 250))
    #img2.data = cv.resize(img2.data, (200, 250))
    
    # Resize the images to 100x125
    #img1.data = cv.resize(img1.data, (100, 125))
    #img2.data = cv.resize(img2.data, (100, 125))
    #plt.imshow(img1.data)
    #plt.show()
    
    #plt.imshow(img2.data)
    #plt.show()
    
    print(f'Image 1 shape: {img1.shape}')
    print(f'Image 2 shape: {img2.shape}')

    w1, h1, c1 = img1.shape     
    w2, h2, c2 = img2.shape
    
    print(f'Image 1 width: {w1}, height: {h1}')
    print(f'Image 2 width: {w2}, height: {h2}')
    
    
    features1 = dff.detect_facial_features(img1)
    features2 = dff.detect_facial_features(img2)
    
    
    
    n = 40
    
    interpolated_image = inverse_distance_interpolation(img1, img2, features1, features2, n, q=4)
    
    #for i in range(len(interpolated_image)):
        
        
        #plt.imshow(interpolated_image[i])
        #plt.show()
        #save_image(interpolated_image[i], os.path.join(output_path, f'interpolated_image_{i}.png'))
    return

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
    number_of_interpolants = 100
    
    img = np.zeros((dim1, dim2, 3))
    
    # Create random set of points and values within the image
    interpolants_x = np.random.randint(0, dim1, (number_of_interpolants, 1))
    interpolants_y = np.random.randint(0, dim2, (number_of_interpolants, 1))
    
    interpolants = np.concatenate((interpolants_x, interpolants_y), axis=1)
    interpolants_value = np.random.randint(0, 255, (number_of_interpolants, 3))
    
    # Set the pixels
    for i in range(interpolants.shape[0]):
        x, y = interpolants[i]
        img[x][y] = interpolants_value[i]
        
    # Save the image
    cv.imwrite(os.path.join(output_path, 'idw-test.png'), img)
    
    w, h, c = img.shape
    
    # Interpolate the rest of the image
    for x in range(w):
        for y in range(h):
            
            # Skip the pixels that are already interpolated
            if(img[x][y].any()):
                continue
            
            # Get the interpolated value
            img[x][y] = inverse_distance_weighting(np.array([x,y], dtype=float), interpolants.astype(float), interpolants_value.astype(float), 4)
            
    # Save the image
    cv.imwrite(os.path.join(output_path, 'idw-test-interpolated.png'), img)

def test_bilinear_sampling():
    
    print("=========== Testing bilinear sampling ===========")
    
    sample_image = np.zeros((2,2,1))
    
    # Set the pixels
    sample_image.data = np.array([[0,1], [2,3]])
    
    """     for row in sample_image.data:
            print(row) """    
            
    outer_corners = np.array([[0,0], [1,0], [0,1], [1,1]])
    pixel_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    
    for i in outer_corners:
        print(f'Sampled value at {i[0]}, {i[1]}: {bilinear_sampling(sample_image.data, i[0], i[1])}')
        
    for i in pixel_centers:
            print(f'Sampled value at {i[0]}, {i[1]}: {bilinear_sampling(sample_image.data, i[0], i[1])}')
                    
    
    mean = np.mean(sample_image.data)
    
    mean_sample = bilinear_sampling(sample_image.data, 1, 1)
    
    print(f'Mean: {mean}, Mean sample: {mean_sample}')
    
    almost_mean = bilinear_sampling(sample_image.data, 1.1, 1)
    
    print(f'Almost mean: {almost_mean}')
    
    resolution = 200

    
    sampled_image = np.zeros((resolution+1, resolution+1))
    
    for h in range(resolution + 1):
        h_t = h / resolution * 2
        
        for w in range(resolution + 1):
            w_t = w / resolution * 2
            
            #print(f'Sampling at {w_t}, {h_t}')
            sampled_image[w][h] = bilinear_sampling(sample_image.data, w_t, h_t)        
        
        
    plt.imshow(sampled_image, cmap='gray')
    plt.show()
    
# Test the different functions
if __name__ == "__main__":
    
    #test_bilinear_sampling()
    #test_inverse_distance_weighting()
    test_inverse_distance_interpolation()