"""
This file contains the code for detecting the face features in an image.
    
"""
from backend.img_utils import *
import numpy as np
import dlib as dl

def detect_facial_features(img: Image) -> np.array:
    """
    This method detects the facial features in an image.
    Args:
        param img: The image to detect the facial features in.
    
    Returns: 
        The facial features in the image as a numpy array of pixel coordinates.
    """
    predictor = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Convert the image to grayscale
    gray = convert_to_grayscale(img)
    
    if(is_valid_image(gray, rgb=False)):
        
        # Detect the face in the image
        detector = dl.get_frontal_face_detector()
        faces = detector(gray.data, 1)
        
        if(len(faces) == 1):
            # Get the facial features
            landmarks = predictor(gray.data, faces[0])
            
            # Convert the facial features to a numpy array
            facial_features = np.zeros((68, 2), dtype=np.uint8)
            for i in range(68):
                facial_features[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
            return facial_features
        else:
            print("No face detected")
    
    
    
    
    