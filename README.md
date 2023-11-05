# Student Workspace for CS4365 Applied Image Processing

| Repository URL                                                                            |
|:------------------------------------------------------------------------------------------|
| https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523hbvmeenen |

# Face morphing

This is a README describing which features I have implemented, my overall approach, how to set up a running environment as well as instructions for how to run my face-morping pipeline (both through a GUI and via the terminal)

## What have I done?
I have implemented all the basic features of the face morphing project. 

Through the Tkinter-based UI a user can load images and detect baseline features. Both images and detected facial features are displayed. Detection is done with Dlib and basic image processing operations (resizing, loading, etc.) are for the most parth done with OpenCV. In the UI the user can add/move/delete feature points using the buttons below the images. Adding a point adds one feature point in each image at the same location, and they can then be moved around seperately. Deleting a point in one image deletes the associated point in the other image.

The user can also tweak face-morphing pipeline settings. The adjustable parameters are:

- Interpolation steps: How many morphing frames are generated
- IDW q parameter: Which Q parameter is used in the IDW procedure. Increasing it will make the overall interpolation "blockier" as the points closest to feature points will have their movement correlate more strongly with the closest point
- GAN refinement steps: How many optimisation steps the StyleGAN2 implementation is allowed to use when projecting the image into latent space. A higher number will produce better results, but is computationally expensive.
- Target resolution: The final output resolution. To prevent bugs and poor resizing, the options are limited in the frontend, to halving or quartering the width and height of the image. 
- Target name: the name of the final video output file.

Feedback along the way is printed to the stdout of the terminal, because frontends are hard and I don't like them.

Once settings are configured correctly the start pipeline button will be enabled and will trigger the Numba accelerated numpy implementation of the interpolation procedure. The overall process is this:

1. Feature points are positionally interpolated along the time axis.

2. For each feature point we store how far it has moved in relation to both images. I call this the feature delta.

3. Using IDW, and the feature deltas, we compute a delta field using IDW to describe where each pixel should backward sample its value in both images.

4. Each pixel samples values from both images with bilinear sampling using their delta values

5. The two samples are linearly interpolated and weighted according to their time-distance to both source images.

This is repeated for each interpolation frame, and they are saved to `/output/interpolation`. These interpolated images are then taken as input into the projection method of StyleGAN2 to produce a closest vector in latent space using an optimisation-based projection method, seemingly implemented according to the Image2StyleGAN paper. Loss is measured with VGG-measured feature loss and MSE of pixel values to preserve features as well as colour values. The latent vectors are used as input into StyleGAN2 which synthesise images based on them. These output are saved tp `/output/projection`

Finally, using ImageIO we save this sequence of files to `/output` as an mp4 file. The results of each procedure can be found in their respective directory as an image sequence or a video file. Additionally, the source images are also saved in `/output` as `img1.jpg` and `img2.jpg`



## Basic features

- [x] 1. Load 2 RBG images of two faces from disk 
    - Implemented in: `src/app.py`, `src/backend_pipeline.py`

- [x] 2. Run a pre-trained face landmark detector on both images.
    - Implemented in `src/detect_face_features.py`

- [x] 3. Allow user to edit/add/remove the landmarks (UI)
    - Implemented in `src/app.py`

- [x] 4. Interpolate the landmark positions and colours from both images (create a morphing sequence for the landmarks alone). Notes: Results may be poor without adding more landmark points in step 3.
    - Implemented in `src/interpolate.py`

- [x] 5. Complete the remaining pixels using Shephard Interpolation (IDW).
    - Implemented in `src/interpolate.py`

- [x] 6. Project the image to a pretrained GAN (e.g. using GAN inversion or Pivotal tuning to improve the image)
    - Implemented in: `src/app.py`, `src/backend_pipeline.py` using modified NVIDIA StyleGAN2-ada implementation. StyleGAN projection implementation found in `src/projector.py` with dependencies in `src/legacy.py` & `src/dnnlib` + `src/torch_utils`

- [x] 7. Repeat steps 4-7 for the entire morphing sequence.
    - Implemented in: `src/app.py`, `src/backend_pipeline.py`

- [x] 8. Save the result (image sequence, video or GIF).
    - Implemented in: `src/app.py`, `src/backend_pipeline.py`


### Implemented using 3rd party lib
|Feature                | 3rd Party lib                              | Affected files (and/or directories)                                  |
|-----------------------|--------------------------------------------|----------------------------------------------------------------------|
|image loading          | OpenCV                                     | Throughout `/src`                                                    |
|Face landmark detection| DLIB                                       | `src/detect_face_features.py`                                        |
|Pretrained GAN         | Nvidia StyleGAN2-ada pytorch implementation| `src/projector.py`, `src/legacy.py`, `src/torch_utils`, `src/dnnlib` |
|Result saver           | ImageIO                                    | Throughout `/src`                                                    |
|UI                     | Tkinter, TTkbootstrap                      | `src/app.py`                                                         |

StyleGAN2 implementation retrieved from: https://github.com/NVlabs/stylegan2-ada-pytorch

## Extended Features

- [ ] Automatically densify the landmarks in Step 3 using consistent triangulation and mesh subdivision in both images (+1.0)
- [ ] Support for objects other than faces (select suitable features) (+1.0)
- [ ] Transfer motion from a video (use an optical flow estimator and move the landmarks based on that) (up to +2.0)


# Environment setup & running the program

## Environment

The program was developed on a desktop:
 - x86 AMD Ryzen 5 5600g
 - 64-bit Ubuntu 22.04.3 LTS machine 
 - 16GB RAM 
 - Nvidia CUDA-enabled RTX3060. 
 
 Some parts were also done on an M2 Macbook air running MacOS Sonoma with 8GB of RAM. I make no guarantees that this thing works on Windows or without a CUDA or MPS enabled device.

Environment specification can be found in environment.yml and activated with

```bash
conda env create -f environment.yml && mkdir output && mkdir rsc/models
conda activate face_morpher
```

2 Machine learning models where used. One for face detection and one for interpolation refinement. The StyleGAN weights should download automatically when running the program, but here are wget commands for both models as needed. If there are problems with the StyleGAN weights then I suggest checking the documentation of the StyleGAN implementation

```bash
#wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat rsc/models/
```

## Running the thing

There are 2 ways to run the image morphing pipeline:

1. Through the UI in `app.py`. 
This allows for selection of images and editing of feature points, but works poorly for very high resolution images (images are not scaled) and places some restrictions on settings for the pipeline.

```bash
python src/app.py
```

2. Through `backend_pipeline.py`. This allows for more flexible pipeline settings, but no method of altering feature points is provided. Pipeline settings are modified at the top of the file.
```bash
python src/backend_pipeline.py
```
