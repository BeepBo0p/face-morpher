"""
This file specifies the UI including the buttons and the layout of the UI.
"""
import tkinter as tk
import tkinter.ttk as ttk
import os


# Paths
project_path = os.getcwd()
data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, 'output')
rsc_path = os.path.join(project_path, 'rsc')
ui_rsc_path = os.path.join(rsc_path, 'ui')





# Initializing the app
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


# Defining the menu bar
# -------------------------------------------------------------- #
menu = tk.Frame(master=app, width=0.1*w, height=h, bg="#FFFFFF")
menu.pack(side=tk.LEFT, fill=tk.Y)

# Defining the options

# Image loading buttons
load_photo_1_button = tk.Button(master=menu, image=load_photo, text="Load Image 1", bg="#FFFFFF", foreground="#000000")
load_photo_2_button = tk.Button(master=menu, image=load_photo, text="Load Image 2", bg="#FFFFFF", foreground="#000000")

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

# Pack the image loading buttons
load_photo_1_button.pack(side=tk.TOP, fill=tk.X)
load_photo_2_button.pack(side=tk.TOP, fill=tk.X)
resolution_label.pack(side=tk.TOP, fill=tk.X)
target_resolution_dropdown.pack(side=tk.TOP, fill=tk.X, pady=(0, 0.2*h))




# Pipeline settings

# Interpolation steps (slider with range 10-100)
interpolation_steps_label = tk.Label(master=menu, text="Interpolation Steps", bg="#FFFFFF", foreground="#000000")
interpolation_steps = tk.IntVar(value=40)
interpolation_steps_slider = tk.Scale(
    master=menu, 
    from_=10, to=100, 
    variable=interpolation_steps, 
    orient=tk.HORIZONTAL, 
    length=0.1*w, 
    bg="#FFFFFF",
    foreground="#000000"
    )

# IDW q parameter (slider with range 1-10)
IDW_q_parameter_label = tk.Label(master=menu, text="IDW q Parameter", bg="#FFFFFF", foreground="#000000")
IDW_q_parameter = tk.IntVar(value=2)
IDW_q_parameter_slider = tk.Scale(
    master=menu, 
    from_=1, to=10, 
    variable=IDW_q_parameter, 
    orient=tk.HORIZONTAL, 
    length=0.1*w, 
    bg="#FFFFFF",
    foreground="#000000"
    )

# GAN refinement steps (slider with range 100-1000, step size 100)
GAN_refinement_steps_label = tk.Label(master=menu, text="GAN Refinement Steps", bg="#FFFFFF", foreground="#000000")
GAN_refinement_steps = tk.IntVar(value=500)
GAN_refinement_steps_slider = tk.Scale(
    master=menu, 
    from_=100, to=1000, 
    variable=GAN_refinement_steps, 
    orient=tk.HORIZONTAL, 
    length=0.1*w, 
    bg="#FFFFFF",
    foreground="#000000"
    )

# Target name (text field)
target_name_label = tk.Label(master=menu, text="Target Name", bg="#FFFFFF", foreground="#000000")
target = tk.StringVar(value="facemorph")
target_name_entry = tk.Entry(master=menu, textvariable=target, bg="#FFFFFF", foreground="#000000")


# Pack the pipeline settings
interpolation_steps_slider.pack(side=tk.TOP, fill=tk.X)
interpolation_steps_label.pack(side=tk.TOP, fill=tk.X,pady=(0,10))
IDW_q_parameter_slider.pack(side=tk.TOP, fill=tk.X)
IDW_q_parameter_label.pack(side=tk.TOP, fill=tk.X, pady=(0,10))
GAN_refinement_steps_slider.pack(side=tk.TOP, fill=tk.X)
GAN_refinement_steps_label.pack(side=tk.TOP, fill=tk.X, pady=(0,10))
target_name_label.pack(side=tk.TOP, fill=tk.X, pady=(0.1*h,0))
target_name_entry.pack(side=tk.TOP, fill=tk.X, pady=(0, 0.2*h))


# Done button
done_button = tk.Button(master=menu, image=done_photo, bg="#FFFFFF")

# Start button
start_pipeline_button = tk.Button(master=menu, image=play_photo, bg="#FFFFFF", state="disabled")

# Pack the start options
start_pipeline_button.pack(side=tk.BOTTOM, fill=tk.X)
done_button.pack(side=tk.BOTTOM, fill=tk.X)




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


add_points_button = tk.Button(master=feature_points_1_frame, image=add_points_photo, bg="#FFFFFF")
add_points_button.pack(side=tk.LEFT)

move_points_button = tk.Button(master=feature_points_1_frame, image=move_photo, bg="#FFFFFF")
move_points_button.pack(side=tk.LEFT)

delete_points_button = tk.Button(master=feature_points_1_frame, image=delete_photo, bg="#FFFFFF")
delete_points_button.pack(side=tk.RIGHT, padx=(0.3*w,0))


# Defining the canvas for image 2
canvas_2 = tk.Canvas(master=canvas_frame, width=0.45*w, height=h, bg="#FFFFFF")
canvas_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

feature_points_2_frame = tk.Frame(
    master=canvas_2, width=0.45*w, height=0.05*h, bg="#FFFFFF")
feature_points_2_frame.pack(side=tk.BOTTOM, fill=tk.X)

add_points_button = tk.Button(master=feature_points_2_frame, image=add_points_photo, bg="#FFFFFF")
add_points_button.pack(side=tk.LEFT)

move_points_button = tk.Button(master=feature_points_2_frame, image=move_photo, bg="#FFFFFF")
move_points_button.pack(side=tk.LEFT)

delete_points_button = tk.Button(master=feature_points_2_frame, image=delete_photo, bg="#FFFFFF")
delete_points_button.pack(side=tk.RIGHT, padx=(0.3*w,0))


# Run the app
# -------------------------------------------------------------- #
app.mainloop()
