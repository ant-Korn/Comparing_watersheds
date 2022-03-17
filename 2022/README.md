# Computational resources comparison of watershed algorithm implementations 

This folder is a set of scripts providing the ability to compare execution time and maximum memory of the watershed algorithm implementations for volumetric images segmentation, which are presented in the following software:
* [IPSDK](https://www.reactivip.com "IPSDK")
* [ITK](https://github.com/InsightSoftwareConsortium/ITK "Insight Segmentation and Registration Toolkit")
* [Mahotas](https://github.com/luispedro/mahotas "Mahotas")
* [Mamba](https://github.com/nicolasBeucher/mamba-image "Mathematical Morphology library Image")
* [MATLAB](https://www.mathworks.com "MATLAB")
* [Octave](https://www.gnu.org/software/octave "Octave")
* [Skimage](https://github.com/scikit-image/scikit-image "scikit-image")
* [SMIL](https://github.com/ensmp-cmm/smil "Simple Morphological Image Library")

### Prerequisites

Python version 3.8 is required to run scripts.
You also need to install all of the previously listed software to be able to call their functions using Python. Some of them (except IPSDK, Mamba, MATLAB, Octave, SMIL) can be installed with other required Python libraries using pip:
```
pip install -r requirements.txt
```
Mamba and SMIL should be installed manually, following the instructions of the documentation: [Mamba docs](http://www.mamba-image.org/doc.html) and [SMIL docs](http://smil.cmm.mines-paristech.fr/wiki/doku.php/download).

IPSDK, MATLAB and Octave should be installed manually too: [IPSDK](https://www.reactivip.com/image-processing/#sdk), [MATLAB](https://www.mathworks.com/products/get-matlab.html) and [Octave](https://www.gnu.org/software/octave/download). 
You also need installed [Image Processing Toolbox](https://www.mathworks.com/products/image.html) for MATLAB and [image](https://octave.sourceforge.io/image) package for Octave.

### Installing

To use these scripts, you need to follow following steps:

```
git clone https://github.com/ant-Korn/Compare_watersheds.git # Clone repo.
cd 2022
```

## Using

We use scripts from folder [plots_of_time_and_memory_versus_image_size](./plots_of_time_and_memory_versus_image_size) for getting plots of time and memory versus tested image size.

We use sctipt [run_watershed.m](./matlab_octave/run_watershed.m) for MATLAB and Octave implementations estimation.
Before script running in Octave we need execute the following command for image package loading:
```
pkg load image
```

And for estimation of watershed segmentation of real huge volumetric images via Python we use Jupyter Notebook [estimation_for_real_images.ipynb](./estimation_for_real_images.ipynb).
