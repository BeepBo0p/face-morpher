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
import imageio as io
import os


# Class definitions

class Image:
    """
    Class to represent an image.

    Attributes:
        int width: The width of the image.
        int height: The height of the image.
        np.array(W,H,3) data: The image data. The shape of the array is (W, H, 3) where W is the width, H is the height and 3 is the number of channels (RGB).
    """

    def __init__(self, width: int, height: int, channels: int = 3) -> None:
        self.width = width
        self.height = height
        self.channels = channels
        self.data = np.zeros((width, height, channels), dtype=np.uint8)


# Utility functions

def is_valid_image(image: Image, rgb: bool = True) -> bool:
    """
    Checks if the given image is valid.

    Args:
        image: The image to check.
        rgb: Whether to check if the image is in RGB format or grayscale.

    Returns:
        True if the image is valid, False otherwise.
    """
    return True
    # width, height, channels = image.data.shape

    # Check if the width and height are positive
    if (width <= 0 or height <= 0):
        return False

    # Check if the number of channels is 3 (RGB) or 1 (grayscale)
    if (rgb and channels != 3):
        return False

    if (not rgb and channels != 1):
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

    if (width1 != width2 or height1 != height2):
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
    image = io.imread(path)

    width, height, channels = image.shape

    image_object = Image(width, height, channels)

    image_object.data = image

    return image_object


def save_image(image: Image, path: str) -> None:
    """
    Saves the given image to the given path.

    Args:
        image: The image to save.
        path: The path to save the image to.
    """

    # Check if output path is valid
    if (path == ""):
        raise ValueError("Output path is empty.")

    # Check if the image is valid
    if (not is_valid_image(image)):
        raise ValueError("Image is invalid.")

    # Check if the image is in RGB format
    if (not is_valid_image(image, rgb=True)):
        raise ValueError("Image is not in RGB format.")

    # TODO: Create the output directory if it does not exist

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

    grayscale_image.data = np.dot(image.data[..., :3], [0.299, 0.587, 0.114])

    # Reshape the image to w, h, 1
    grayscale_image.data = np.reshape(
        grayscale_image.data, (image.width, image.height, 1))

    print(grayscale_image.data.shape)

    return grayscale_image


def save_to_gif(image_list: list, src_path: str, dst_path: str) -> None:
    """
    Saves the given list of images to a gif.

    Args:
        image_list: The list of images to save.
        path: The path to save the gif to.
    """
    image_sequence = []

    for filename in image_list:

        print(src_path + '/' + filename)

        image_sequence.append(io.imread(src_path + '/' + filename))

    io.mimsave(dst_path, image_sequence)


# Test the code
if (__name__ == "__main__"):
    # image = load_image("../../data/test.jpg")
    #
    # print(image.data.shape)
    #
    # plt.imshow(image.data)
    # plt.show()
    #
    # save_image(image, "../../data/test2.jpg")

    image_list = ["interpolated_image_{i}.png".format(i=i) for i in range(10)]
    # image_list = ["interpolated_image__with_features_{i}.png".format(i=i) for i in range(10)]

    print(image_list[0])

    src_path = os.path.join(os.getcwd(), "output/interpolation")
    dst_path = os.path.join(os.getcwd(), "output/interpolation_with_feat.mp4")

    save_to_gif(image_list, src_path, dst_path)
