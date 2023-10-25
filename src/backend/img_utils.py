"""
This file contains the code for loading the image and converting it into a standard format.

While working on the MVP we will assume that the images are already in a standard format.

That is to say: dim(A) = dim(B) = (W, H, 3) 

where W, H are the width and height of the image respectively, and 3 is the number of channels (RGB).

At a later stage we will maybe add code to convert the images to a standard format, 
but this is a sort of pre-processing step and can be done in an outside program.
"""
import numpy as np
import matplotlib.pyplot as plt

# Class definitions

class Image:
    """
    Class to represent an image.
    
    Attributes:
        int width: The width of the image.
        int height: The height of the image.
        np.array(W,H,3) data: The image data. The shape of the array is (W, H, 3) where W is the width, H is the height and 3 is the number of channels (RGB).
    """
    
    def __init__(self, width: int, height: int, channels: int=3) -> None:
        self.width = width
        self.height = height
        self.channels = channels
        self.data = np.zeros((width, height, channels), dtype=np.uint8)
        

# Utility functions

def is_valid_image(image: Image, rgb: bool=True) -> bool:
    """
    Checks if the given image is valid.
    
    Args:
        image: The image to check.
        rgb: Whether to check if the image is in RGB format or grayscale.
        
    Returns:
        True if the image is valid, False otherwise.
    """
    width, height, channels = image.data.shape
    
    # Check if the width and height are positive
    if(width <= 0 or height <= 0):
        return False
    
    # Check if the number of channels is 3 (RGB) or 1 (grayscale)
    if(rgb and channels != 3):
        return False
    
    if(not rgb and channels != 1):
        return False
    
    return True
    
    
    return True
    
def is_same_size(image1: Image, image2: Image) -> bool:
    """
    Checks if the given images are the same size.
    
    Args:
        image1: The first image.
        image2: The second image.
        
    Returns:
        True if the images are the same size, False otherwise.
    """
    width1, height1, _ = image1.data.shape
    width2, height2, _ = image2.data.shape
    
    if(width1 != width2 or height1 != height2):
        return False
    
    return True

def load_image(path: str) -> Image:
    """
    Loads an image from the given path and returns an Image object.
    
    Args:
        path: The path to the image.
        
    Returns:
        An Image object.
    """
    image = plt.imread(path)
    
    width, height, _ = image.shape
    
    image_object = Image(width, height)
    
    image_object.data = image
    
    return image_object

def save_image(image: Image, path: str) -> None:
    """
    Saves the given image to the given path.
    
    Args:
        image: The image to save.
        path: The path to save the image to.
    """
    plt.imsave(path, image.data)
    
def show_image(image: Image) -> None:
    """
    Shows the given image.
    
    Args:
        image: The image to show.
    """
    plt.imshow(image.data)
    plt.show()

def convert_to_grayscale(image: Image) -> Image:
    """
    Converts the given image to grayscale.
    
    Args:
        image: The image to convert.
        
    Returns:
        The converted image.
    """
    grayscale_image = Image(image.width, image.height)
    
    grayscale_image.data = np.dot(image.data[...,:3], [0.299, 0.587, 0.114])
    
    return grayscale_image

# Test the code
if(__name__ == "__main__"):
    image = load_image("../../data/test.jpg")
    
    print(image.data.shape)
    
    plt.imshow(image.data)
    plt.show()
    
    save_image(image, "../../data/test2.jpg")
    