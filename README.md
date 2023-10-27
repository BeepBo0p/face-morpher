# Student Workspace for CS4365 Applied Image Processing

Use this workspace to solve your assignments and projects during the course CS4365 Applied Image Processing.

**Commit often** (at least once a day if you make any changes) and provide **meaningful commit messages** to document your progress while solving the programming tasks.

# Face morphing

## Basic features


- [ ] 1. Load 2 RBG images of two faces from disk 
- [ ] 2. Run a pre-trained face landmark detector on both images.
- [ ] 3. Allow user to edit/add/remove the landmarks (UI)
- [ ] 4. Interpolate the landmark positions and colours from both images (create a morphing sequence for the landmarks alone). Notes: Results may be poor without adding more landmark points in step 3.
- [ ] 5. Complete the remaining pixels using Shephard Interpolation (IDW).
- [ ] 6. Project the image to a pretrained GAN (e.g. using GAN inversion or Pivotal tuning to improve the image)
- [ ] 7. Repeat steps 4-7 for the entire morphing sequence.
- [ ] 8. Save the result (image sequence, video or GIF).

### Can be implemented using 3rd party lib
- image loading
- Face landmark detection
- Pretrained GAN
- GIF saver

## Extended Features

- [ ] Automatically densify the landmarks in Step 3 using consistent triangulation and mesh subdivision in both images (+1.0)
- [ ] Support for objects other than faces (select suitable features) (+1.0)
- [ ] Transfer motion from a video (use an optical flow estimator and move the landmarks based on that) (up to +2.0)


## Clarification: Point interpolation

A direct interpolation of colours will lead to a suboptimal quality.
The colours should instead be interpolated indirectly.
