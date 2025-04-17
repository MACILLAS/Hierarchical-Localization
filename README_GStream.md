# HLOC for Simultaneous Gaussian Splatting

The purpose of this fork is to use HLOC pipeline to support the Gaussian Splatting inspection application.

Changes:

-[x] Modify reconstruction.py to use single PINHOLE camera model instead of default Simple Radial
-[x] Add pipeline for sparse_reconstruction using superpoint + superglue
-[x] Pre-processor to flip and rename input images
-[x] Triangulation with pre-calculated extrinsic parameters

## Pre-Processing

* Poly-CAM saves images as landscape... To handel this we add the portrait flag

```bash
# This script saves images in "out_images" folder in proj directory... 
python ./datasets/image_pre_process.py --proj st_1 --portrait
```

* Remove the original images directory and rename "out_images" to "images".

* Repeat for other proj scans as necessary.

* (Optional) Create an additional proj, proj_comb, composed of a union of images from both projs

## SfM

* Use the pipeline_st_1.ipynb to start SfM reconstruction (Change paths as necessary)
  * This will yield the proj folder in ./outputs/proj/


./outputs/proj/sfm/ --> contains extracted features (global and local) and matches.
These will be used to do reconstruction and triangulation.

./outputs/proj/sfm/proj/sfm_superpoint+superglue --> contains the outputs of the sparse_reconstruction.
(i.e., cameras.bin, database.db, images.bin, points3D.bin) These can be opened by Colmap.

I recommend saving at least a copy of the images.txt file which will be useful for our GS_Stream application.

## Triangulate w/ Known Camera Poses

https://github.com/cvg/Hierarchical-Localization/issues/222

In the previous (optional step) we created a proj_comb dataset. 
The purpose is to allow us to triangulate an SfM model with known poses (which are precalculated)

Assuming you have created a SfM reconstruction with proj_comb images...

See parse_st_comb.py to create empty, pose initialized model...

* Use the pipeline_st_triangulation.ipynb which leverage the extracted features and matches for the subset of proj_comb