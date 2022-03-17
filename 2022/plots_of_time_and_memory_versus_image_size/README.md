### Installing

To use these scripts, you need to follow following steps:

```
git clone https://github.com/ant-Korn/Compare_watersheds.git # Clone repo.
cd 2022/plots_of_time_and_memory_versus_image_size
make init # Creating directory structure, series of the test images. 
```

## Using

[Makefile](./Makefile) provide various commands with default parameters for comparison via all software except MATLAB and Octave:
```
make init # Creating directory structure, series of the test images. 
make compare_all_3D # Comparing execution time and maximum memory for 3D case with and
                    # without a watershed lines (WL) construction.
make gen_3D # Generating 3D test image.
make compare_3D_time # Comparing execution time for 3D case without WL construction.
make compare_3D_time_WL # Comparing execution time for 3D case with WL construction.
make compare_3D_mem # Comparing maximum memory for 3D case without WL construction.
make compare_3D_mem_WL # Comparing maximum memory for 3D case with WL construction.
make plot_all # Creating plots based on all comparisons.
make plots_time # Creating plots based on time comparisons.
make plots_mem # Creating plots based on memory comparisons.
make clean    #
make clean_3D # Removing generated images.
```

You can execute the comparisons separately or together (these operations can take a long time). The results of the comparison will be stored in directory *$(LOGS_FOLDER)* (default *logs_watershed/*) under the names *3D_proc_{mem,time}[_WL].csv* for different cases (memory/time, without/with WL).
Then you can plot graphs based on this comparisons. The resulting graphs will be saved in the PDF format to directory *$(PLOTS_FOLDER)* (default *plots/*) under the names *imgname[memory][_WL].pdf* for different cases (time/memory, without/with WL).
