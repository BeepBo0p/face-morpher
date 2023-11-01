"""
This file specifies the UI including the buttons and the layout of the UI.
"""
# Local imports
import backend_pipeline as bkp

# Functional imports
import os
import cv2 as cv
import numpy as np

# UI imports
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as fd

# -------------------------------------------------------------- #
# Global variables                                               #
# -------------------------------------------------------------- #

# Paths
# -------------------------------------------------------------- #
project_path = os.getcwd()
data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, 'output')
rsc_path = os.path.join(project_path, 'rsc')
ui_rsc_path = os.path.join(rsc_path, 'ui')

# Images and feature points
# -------------------------------------------------------------- #
image_1 = None
image_2 = None
feature_points_1 = []
feature_points_2 = []




# -------------------------------------------------------------- #
# Defining the functions for the buttons                         #
# -------------------------------------------------------------- #

# Image loading buttons
# -------------------------------------------------------------- #
def open_image_1():
    """
    Opens an image from the file system and displays it on the canvas.
    """
    global img_1_path
    
    path = fd.askopenfilename()
    
    img_1_path.set(path)
    
    print(f'Image 1 path set to {path}')
    
def open_image_2():
    """
    Opens an image from the file system and displays it on the canvas.
    """
    global img_2_path
    
    path = fd.askopenfilename()
    
    img_2_path.set(path)    

    print(f'Image 2 path set to {path}')
        
def validate_and_load_images():
    """
    Validates the images and loads them into the canvas.
    """
    global image_1
    global image_2
    global img_1_path
    global img_2_path
    
    # Check if the image paths are valid
    
    # Load the images into memory
    
    # Check if the images are (approximately) the same ratio
    
    # Find the smaller image
    
    # Resize the larger image to the size of the smaller image
    
    # Display the images on the canvases
    pass
    
    
    

# Feature point buttons
# -------------------------------------------------------------- #
def add_points_1():
    """
    Adds feature points to the image.
    """
    global feature_points_1
    global feature_points_2
    
    global image_1
    global image_2
    
    # Get nearest pixel coordinates at mouse click
    
    # Add the pixel coordinates to the feature points list
    
    # Add a corresponding point to the other image's feature points list
    
    # Draw the feature points on the canvas for both images
    pass

def add_points_2():
    """
    Adds feature points to the image.
    """
    global feature_points_1
    global feature_points_2
    
    global image_1
    global image_2
    
    # Get nearest pixel coordinates at mouse click
    
    # Add the pixel coordinates to the feature points list
    
    # Add a corresponding point to the other image's feature points list
    
    # Draw the feature points on the canvas for both images
    pass

def move_points_1():
    """
    Moves feature points on the image.
    """
    global feature_points_1
    global image_1
    
    # Get nearest feature point to mouse click
    
    # Click at new position
    
    # Update the feature point coordinates
    
    # Update the canvas
    
    pass

def move_points_2():
    """
    Moves feature points on the image.
    """
    global feature_points_2
    global image_2
    
    # Get nearest feature point to mouse click
    
    # Click at new position
    
    # Update the feature point coordinates
    
    # Update the canvas
    
    pass

def delete_points_1():
    """
    Deletes feature points on the image.
    """
    global image_1
    global image_2
    
    global feature_points_1
    global feature_points_2
    
    # Get nearest feature point to mouse click
    
    # Find the index of the feature point in the feature points list
    
    # Delete the feature point from the feature points list
    
    # Delete the corresponding feature point from the other image's feature points list
    
    # Update the canvas
    pass

def delete_points_2():
    """
    Deletes feature points on the image.
    """
    global image_1
    global image_2
    
    global feature_points_1
    global feature_points_2
    
    # Get nearest feature point to mouse click
    
    # Find the index of the feature point in the feature points list
    
    # Delete the feature point from the feature points list
    
    # Delete the corresponding feature point from the other image's feature points list
    
    # Update the canvas
    pass


# Pipeline buttons
# -------------------------------------------------------------- #
def validate_settings():
    """
    Validates the settings and enables the start button.
    """
    global image_1
    global image_2
    global feature_points_1
    global feature_points_2
    
    global interpolation_steps
    global IDW_q_parameter
    global GAN_refinement_steps
    
    global target_resolution
    global start_pipeline_button
    
    
    if interpolation_steps.get() <= 1:
        raise ValueError("Interpolation steps must be greater than 1.")
    
    if interpolation_steps.get() % 3 == 0:
        raise ValueError("Bug in bilinear sampling. Interpolation steps must not be divisible by 3.")
    
    if IDW_q_parameter.get() <= 0:
        raise ValueError("IDW q parameter must be greater than 0.")
    
    if GAN_refinement_steps.get() <= 0:
        raise ValueError("GAN refinement steps must be greater than 0.")
    
    if target_resolution.get() not in ["original", "half", "quarter"]:
        raise ValueError("Target resolution must be one of the following: original, half, quarter.")
    
    global start_pipeline_button
    
    start_pipeline_button.config(state="normal")
    
    print("Pipeline settings are in valid configuration. Start button enabled.")
    
    pass

def start_pipeline():
    """
    Starts the pipeline.
    """
    
    global interpolation_steps
    global IDW_q_parameter
    global GAN_refinement_steps
    
    global target_resolution
    global target
    
    # Set resolution based on target resolution setting
    case = target_resolution.get()
    
    match case:
        
        case "original":
            pass
        
        case "half":
            image_1 = cv.resize(image_1, (image_1.shape[0]//2, image_1.shape[1]//2))
            image_2 = cv.resize(image_2, (image_2.shape[0]//2, image_2.shape[1]//2))
            
            feature_points_1 = np.array(feature_points_1) // 2
            feature_points_2 = np.array(feature_points_2) // 2
            
        case "quarter":
            image_1 = cv.resize(image_1, (image_1.shape[0]//4, image_1.shape[1]//4))
            image_2 = cv.resize(image_2, (image_2.shape[0]//4, image_2.shape[1]//4))
            
            feature_points_1 = np.array(feature_points_1) // 4
            feature_points_2 = np.array(feature_points_2) // 4
            
    print(f"Initiating pipeline with following settings:")
    print(f"Interpolation steps: {interpolation_steps.get()}")
    print(f"IDW q parameter: {IDW_q_parameter.get()}")
    print(f"GAN refinement steps: {GAN_refinement_steps.get()}")
    print(f"Target resolution: {target_resolution.get()}")
    print(f"Target name: {target.get()}")
    
    
    bkp.morph_faces(
        img1=image_1,
        img2=image_2,
        facial_features_list=[feature_points_1, feature_points_2],
        interpolation_steps=interpolation_steps.get(), 
        idw_q_parameter=IDW_q_parameter.get(), 
        gan_refinement_steps=GAN_refinement_steps.get(), 
        output_path=output_path,
        output_name=target.get() + '.mp4'
    )
    
    print("Pipeline finished.")
    print("Output saved to: ", os.path.join(output_path, target.get() + '.mp4'))
    pass



# -------------------------------------------------------------- #
# Define and pack the app                                        #
# -------------------------------------------------------------- #
app = tk.Tk()

sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()

print("Screen Width: ", sw)
print("Screen Height: ", sh)

screen_coverage = 0.6

w, h = int(screen_coverage*sw), (screen_coverage*sh)


# Photos
# -------------------------------------------------------------- #
load_photo = tk.PhotoImage(file=os.path.join(ui_rsc_path, "add_photo.png"))
done_photo = tk.PhotoImage(file=os.path.join(ui_rsc_path, "done.png"))
play_photo = tk.PhotoImage(file=os.path.join(ui_rsc_path, "play.png"))
move_photo = tk.PhotoImage(file=os.path.join(ui_rsc_path, "move.png"))
delete_photo = tk.PhotoImage(file=os.path.join(ui_rsc_path, "delete.png"))
add_points_photo = tk.PhotoImage(file=os.path.join(ui_rsc_path, "add.png"))

# State variables
# -------------------------------------------------------------- #
img_1_path = tk.StringVar(value="")
img_2_path = tk.StringVar(value="")
interpolation_steps = tk.IntVar(value=40)
IDW_q_parameter = tk.IntVar(value=2)
GAN_refinement_steps = tk.IntVar(value=500)


# Defining the menu bar
# -------------------------------------------------------------- #
menu = tk.Frame(master=app, width=0.1*w, height=h, bg="#FFFFFF")
menu.pack(side=tk.LEFT, fill=tk.Y)



# Image loading buttons
# -------------------------------------------------------------- #

load_photo_1_button = tk.Button(
    master=menu, 
    image=load_photo, 
    text="Load Image 1", 
    textvariable=img_1_path,
    bg="#FFFFFF", 
    foreground="#000000",
    command=open_image_1
    )

load_photo_2_button = tk.Button(
    master=menu, 
    image=load_photo, 
    text="Load Image 2", 
    bg="#FFFFFF", 
    foreground="#000000",
    command=open_image_2
    )

# Done button
load_button = tk.Button(
    master=menu, 
    image=done_photo, 
    bg="#FFFFFF",
    state="disabled",
    command=validate_and_load_images
    )



# Pack the image loading buttons
load_photo_1_button.pack(side=tk.TOP, fill=tk.X)
load_photo_2_button.pack(side=tk.TOP, fill=tk.X)
load_button.pack(side=tk.TOP, fill=tk.X, pady=(0, 0.2*h))




# Pipeline settings
# -------------------------------------------------------------- #

# Interpolation steps (slider with range 10-100)
interpolation_steps_label = tk.Label(master=menu, text="Interpolation Steps", bg="#FFFFFF", foreground="#000000")
interpolation_steps_slider = tk.Scale(
    master=menu, 
    from_=10, 
    to=100, 
    variable=interpolation_steps, 
    orient=tk.HORIZONTAL, 
    length=0.1*w, 
    bg="#FFFFFF",
    foreground="#000000"
    )

# IDW q parameter (slider with range 1-10)
IDW_q_parameter_label = tk.Label(master=menu, text="IDW q Parameter", bg="#FFFFFF", foreground="#000000")
IDW_q_parameter_slider = tk.Scale(
    master=menu, 
    from_=1,
    to=10, 
    variable=IDW_q_parameter, 
    orient=tk.HORIZONTAL, 
    length=0.1*w, 
    bg="#FFFFFF",
    foreground="#000000"
    )

# GAN refinement steps (slider with range 100-1000, step size 100)
GAN_refinement_steps_label = tk.Label(master=menu, text="GAN Refinement Steps", bg="#FFFFFF", foreground="#000000")
GAN_refinement_steps_slider = tk.Scale(
    master=menu, 
    from_=100, 
    to=1000, 
    variable=GAN_refinement_steps, 
    orient=tk.HORIZONTAL, 
    length=0.1*w, 
    bg="#FFFFFF",
    foreground="#000000"
    )



# Pack the pipeline settings
interpolation_steps_slider.pack(side=tk.TOP, fill=tk.X)
interpolation_steps_label.pack(side=tk.TOP, fill=tk.X,pady=(0,10))
IDW_q_parameter_slider.pack(side=tk.TOP, fill=tk.X)
IDW_q_parameter_label.pack(side=tk.TOP, fill=tk.X, pady=(0,10))
GAN_refinement_steps_slider.pack(side=tk.TOP, fill=tk.X)
GAN_refinement_steps_label.pack(side=tk.TOP, fill=tk.X, pady=(0,10))

# Target settings and start button
# -------------------------------------------------------------- #


# Image target resolution (dropdown menu with 3 options: original, half, quarter)
resolution_label = tk.Label(master=menu, text="Target Resolution", bg="#FFFFFF", foreground="#000000")
target_resolution = tk.StringVar(value="original")
target_resolution_options = ["original", "half", "quarter"]
target_resolution_dropdown = ttk.Combobox(
    master=menu, 
    textvariable=target_resolution, 
    values=target_resolution_options, 
    foreground="#000000",
    background="#FFFFFF",
    state="readonly", 
    )

# Target name (text field)
target_name_label = tk.Label(master=menu, text="Target Name", bg="#FFFFFF", foreground="#000000")
target = tk.StringVar(value="facemorph")
target_name_entry = tk.Entry(
    master=menu, 
    textvariable=target, 
    bg="#FFFFFF", 
    foreground="#000000",
    )


# Done button
done_button = tk.Button(
    master=menu, 
    image=done_photo, 
    bg="#FFFFFF",
    command=validate_settings,
    )

# Start button
start_pipeline_button = tk.Button(
    master=menu, 
    image=play_photo, 
    bg="#FFFFFF", 
    state="disabled",
    command=start_pipeline,
    )

# Pack the start options
start_pipeline_button.pack(side=tk.BOTTOM, fill=tk.X)
done_button.pack(side=tk.BOTTOM, fill=tk.X)
target_name_entry.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
target_name_label.pack(side=tk.BOTTOM, fill=tk.X, )
target_resolution_dropdown.pack(side=tk.BOTTOM, fill=tk.X, )
resolution_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(0.2*h,0))




# Defining the canvases for image display
# -------------------------------------------------------------- #
canvas_frame = tk.Frame(master=app, width=0.9*w, height=h, bg="#808080")
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Defining the canvas for image 1
canvas_1 = tk.Canvas(master=canvas_frame, width=0.45*w, height=h, bg="#FFFFFF")
canvas_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add feature points button grid
feature_points_1_frame = tk.Frame(
    master=canvas_1, width=0.45*w, height=0.05*h, bg="#FFFFFF")
feature_points_1_frame.pack(side=tk.BOTTOM, fill=tk.X)


add_points_button = tk.Button(
    master=feature_points_1_frame, 
    image=add_points_photo, 
    bg="#FFFFFF",
    command=add_points_1,
    )

move_points_button = tk.Button(
    master=feature_points_1_frame, 
    image=move_photo, 
    bg="#FFFFFF",
    command=move_points_1,
    )

delete_points_button = tk.Button(
    master=feature_points_1_frame,
    image=delete_photo,
    bg="#FFFFFF",
    command=delete_points_1,
    )

# Pack the feature point buttons
add_points_button.pack(side=tk.LEFT)
move_points_button.pack(side=tk.LEFT)
delete_points_button.pack(side=tk.RIGHT, padx=(0.3*w,0))





# Adding a separator
separator = ttk.Separator(master=canvas_frame, orient=tk.VERTICAL)
separator.pack(side=tk.LEFT, fill=tk.Y, )#padx=(0.005*w,0.005*w))






# Defining the canvas for image 2
canvas_2 = tk.Canvas(master=canvas_frame, width=0.45*w, height=h, bg="#FFFFFF")
canvas_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

feature_points_2_frame = tk.Frame(
    master=canvas_2, width=0.45*w, height=0.05*h, bg="#FFFFFF")
feature_points_2_frame.pack(side=tk.BOTTOM, fill=tk.X)

add_points_button = tk.Button(
    master=feature_points_2_frame, 
    image=add_points_photo,
    bg="#FFFFFF",
    command=add_points_2
    )

move_points_button = tk.Button(
    master=feature_points_2_frame,
    image=move_photo,
    bg="#FFFFFF",
    command=move_points_2
    )

delete_points_button = tk.Button(
    master=feature_points_2_frame, 
    image=delete_photo, 
    bg="#FFFFFF",
    command=delete_points_2
    )

# Pack the feature point buttons
add_points_button.pack(side=tk.LEFT)
move_points_button.pack(side=tk.LEFT)
delete_points_button.pack(side=tk.RIGHT, padx=(0.3*w,0))


# Run the app
# -------------------------------------------------------------- #
app.mainloop()
