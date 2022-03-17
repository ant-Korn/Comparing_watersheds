# Computational resources comparison of watershed algorithm implementations 

This project is a set of scripts providing the ability to compare execution time and maximum memory of the watershed algorithm implementations, which are presented in the following libraries:
* [ITK](https://github.com/InsightSoftwareConsortium/ITK "Insight Segmentation and Registration Toolkit")
* [OpenCV](https://github.com/opencv/opencv "Open Source Computer Vision Library")
* [Mahotas](https://github.com/luispedro/mahotas "Mahotas")
* [Mamba](https://github.com/nicolasBeucher/mamba-image "Mathematical Morphology library Image")
* [Skimage](https://github.com/scikit-image/scikit-image "scikit-image")
* [SMIL](https://github.com/ensmp-cmm/smil "Simple Morphological Image Library")

### Prerequisites

For run scripts you need to install [ImageMagick](https://github.com/ImageMagick/ImageMagick "ImageMagick"), that provide scaling of initial images for test series generation.
Also, Python version 3.6 is required to run scripts.
You also need to install all of the previously listed libraries to be able to call their functions using Python. Some of them (except Mamba and SMIL) can be installed with other required Python libraries using pip:
```
pip install -r requirements.txt
```
Mamba and SMIL should be installed manually, following the instructions of the documentation: [Mamba docs](http://www.mamba-image.org/doc.html) and [SMIL docs](http://smil.cmm.mines-paristech.fr/wiki/doku.php/download).

### Installing

To use these scripts, you need to follow following steps:

```
git clone https://github.com/ant-Korn/Compare_watersheds.git # Clone repo.
cd 2018
make init # Creating directory structure, series of the test images. 
```

## Using

[Makefile](./Makefile) provide various commands with default parameters for comparison:
```
make init # Creating directory structure, series of the test images. 
make compare_all_2D # Comparing execution time and maximum memory for 2D case with and
                    # without a watershed lines construction.
make compare_all_3D # Comparing execution time and maximum memory for 3D case with and
                    # without a watershed lines (WL) construction.
make gen_imgs # Generating series of the test 2D images based on images in $(SOURCE_FOLDER)
make gen_3D # Generating 3D test image.
make compare_2D_time # Comparing execution time for 2D case without WL construction.
make compare_2D_time_WL # Comparing execution time for 2D case with WL construction.
make compare_3D_time # Comparing execution time for 3D case without WL construction.
make compare_3D_time_WL # Comparing execution time for 3D case with WL construction.
make compare_2D_mem # Comparing maximum memory for 2D case without WL construction.
make compare_2D_mem_WL # Comparing maximum memory for 2D case with WL construction.
make compare_3D_mem # Comparing maximum memory for 3D case without WL construction.
make compare_3D_mem_WL # Comparing maximum memory for 3D case with WL construction.
make plot_all # Creating plots based on all comparisons.
make plots_time # Creating plots based on time comparisons.
make plots_mem # Creating plots based on memory comparisons.
make clean    #
make clean_2D #
make clean_3D # Removing generated images.
```

You can execute the comparisons separately or together (these operations can take a long time). The results of the comparison will be stored in directory *$(LOGS_FOLDER)* (default *logs_watershed/*) under the names *[3D_]proc_{mem,time}[_WL].csv* for different cases (2D/3D, memory/time, without/with WL).
Then you can plot graphs based on this comparisons. The resulting graphs will be saved in the PDF format to directory *$(PLOTS_FOLDER)* (default *plots/*) under the names *imgname[memory][_WL].pdf* for different cases (time/memory, without/with WL).
