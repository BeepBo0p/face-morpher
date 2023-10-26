"""
This file contains the code for detecting the face features in an image.
    
"""
from backend.img_utils import *
import numpy as np
import dlib as dl
import os

def detect_facial_features(img: Image) -> np.array:
    """
    This method detects the facial features in an image.
    Args:
        param img: The image to detect the facial features in.
    
    Returns: 
        The facial features in the image as a numpy array of pixel coordinates.
    """
    
    project_path = os.getcwd()
    

    model_path = os.path.join(project_path, "rsc/models/shape_predictor_68_face_landmarks.dat")
    
    predictor = dl.shape_predictor(model_path)
    
    #TODO: Clean up and verify it works on non-grayscale images
    # Convert the image to grayscale
    #gray = convert_to_grayscale(img)
    gray = img
    
    if(is_valid_image(gray, rgb=True)):
        
        # Detect the face in the image
        detector = dl.get_frontal_face_detector()
        faces = detector(gray.data, 1)
        
        print("Number of faces detected: {}".format(len(faces)))
        
        if(len(faces) == 1):
            # Get the facial features
            landmarks = predictor(gray.data, faces[0])
            
            # Show the image with the facial features
            plt.imshow(gray.data, cmap="gray")
            for i in range(68):
                print(f"landmark {i}: {landmarks.part(i)}")
                landmark = landmarks.part(i)
                plt.plot(landmarks.part(i).x, landmarks.part(i).y, "ro")
            plt.show()
            
            # Convert the facial features to a numpy array
            facial_features = np.zeros((68, 2))
            for i in range(68):
                facial_features[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
            return facial_features
        else:
            print("No face detected")
    
    
    
    
def test_face_detection():
    
    # Run from project root
    project_path = os.getcwd()
    
    data_path = os.path.join(project_path, "data")
    
    
    img_path = os.path.join(data_path, "test.jpg")
    dorde_path = os.path.join(data_path, "dorde.jpg")
    jørgen_path = os.path.join(data_path, "jørgen.jpg")
    
    img = load_image(img_path)
    dorde = load_image(dorde_path)
    jørgen = load_image(jørgen_path)
    
    features = detect_facial_features(img)
    dorde_features = detect_facial_features(dorde)
    jørgen_features = detect_facial_features(jørgen)    
    


if __name__ == "__main__":
    test_face_detection()