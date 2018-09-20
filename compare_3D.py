import os
from sys import path
import argparse
from tools import create_data_frame_3D, Image_3D, update_dataframe, gen_sizes_3D, image_generator
import math
from gen_3D import rad, indent, filename


parser = argparse.ArgumentParser(description='Compare watershed segmentation time and memory of 3D images.')
parser.add_argument("-wl", "--watershed_lines", help="Construct watershed lines.", action='store_true')
parser.add_argument("--save", help="Save results of segmentation.", action='store_true')
parser.add_argument("-d", "--dir", help="Images directory.")
parser.add_argument("--resource", help="Can evaluate time and memory.")
parser.add_argument("--logs_dir", help="Directory with result of segmentation.")
args = parser.parse_args()

resource = args.resource

NEED_SAVE = args.save

if args.logs_dir is not None:
    LOGS_FOLDER = os.path.join(args.logs_dir, "3D")
else:   
    LOGS_FOLDER = 'logs_watershed/3D'

NEED_WL = args.watershed_lines

if args.dir is not None:
    IMGS_FOLDER = args.dir
else:
    IMGS_FOLDER = 'source_imgs_3D'

INIT_SIZE = 64
INIT_MIN_DIST = 0

sizes_list = gen_sizes_3D(41, 400000, INIT_SIZE)

iterables = [['skimage', 'mahotas', 'smil', 'mamba', 'itk'],
             ['img_name', 'size', 'time']]
df = create_data_frame_3D(iterables, filename, sizes_list)


if __name__ == '__main__':
    path_initial_image = os.path.join(IMGS_FOLDER, filename+'{0:04}.png')

    for cur_img in image_generator(IMGS_FOLDER, filename, INIT_SIZE, sizes_list):
        print("Process", filename, 'size'+str(cur_img.size))

        cur_img.init_label(rad, indent)

        # SKIMAGE PROCESSING
        ALGO = "skimage"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])

        update_dataframe(df, cur_img, ALGO, resource)

        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER,ALGO)            


        # SMIL PROCESSING
        ALGO = "smil"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])

        update_dataframe(df, cur_img, ALGO, resource)

        # Display output
        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER, ALGO)

        # MAMBA PROCESSING

        ALGO = "mamba"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])

        update_dataframe(df, cur_img, ALGO, resource)

        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER,ALGO)            

        # MAHOTAS PROCESSING
        ALGO = "mahotas"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])

        update_dataframe(df, cur_img, ALGO, resource)

        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER,ALGO)            

        # ITK PROCESSING
        ALGO = "itk"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])
        
        update_dataframe(df, cur_img, ALGO, resource)

        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER,ALGO)    

    if resource == 'time':
        if NEED_WL:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "3D_proc_time_WL.csv"))
        else:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "3D_proc_time.csv"))
    elif resource == 'memory':
        if NEED_WL:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "3D_proc_mem_WL.csv"))
        else:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "3D_proc_mem.csv"))
