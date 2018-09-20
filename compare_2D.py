import os
from sys import path
import argparse
from tools import create_data_frame, Image_2D, update_dataframe
import cv2

# Watershed segmentations
''' skimage.morphology.watershed,
    cv2.watershed,
    smilPython
    mamba.watershedSegment
    mahotas.cwatershed
    itk
'''

parser = argparse.ArgumentParser(description='Compare watershed segmentation time and memory of 2D images.')
parser.add_argument("-wl", "--watershed_lines", help="Construct watershed lines.", action='store_true')
parser.add_argument("--save", help="Save results of segmentation.", action='store_true')
parser.add_argument("-d", "--dir", help="Images directory.")
parser.add_argument("--resource", help="Can evaluate time and memory.")
parser.add_argument("--logs_dir", help="Directory with result of segmentation.")
args = parser.parse_args()

resource = args.resource

NEED_SAVE = args.save

if args.logs_dir is not None:
    LOGS_FOLDER = args.logs_dir
else:   
    LOGS_FOLDER = 'logs_watershed'

NEED_WL = args.watershed_lines

if args.dir is not None:
    IMGS_FOLDER = args.dir
else:
    IMGS_FOLDER = 'dest_imgs'

INIT_SIZE = 512
INIT_MIN_DIST = 0
   
imgs = os.listdir(IMGS_FOLDER)
imgs.sort(key=lambda x: int(x.split('.',1)[0].rsplit('_',1)[1]))

image_init_min_dist = {'board': 20,
                       'circles': 6,
                       'coins': 35,
                       'maze':13,
                       'fruits': 15}

image_need_invert = ['circles', 'coins']

iterables = [['skimage', 'opencv', 'mahotas', 'smil', 'mamba', 'itk'],
             ['img_name', 'size', 'time']]
df = create_data_frame(imgs, iterables)


if __name__ == '__main__':
    for filename in imgs:
        cur_img = Image_2D(IMGS_FOLDER, filename, INIT_SIZE)
        INIT_MIN_DIST = image_init_min_dist[cur_img.cluster]

        print("Process", cur_img.filename)

        #if NEED_SAVE:
        #    cv2.pyrMeanShiftFiltering(cur_img.img, int(10*cur_img.mul_size), 10)

        cur_img.init_label(INIT_MIN_DIST, image_need_invert)
        if cur_img.cluster == 'maze':
            cur_img.markers[:] = 0
            cur_img.markers[int(490*cur_img.mul_size),int(500*cur_img.mul_size)] = 1

    # SKIMAGE PROCESSING
        ALGO = "skimage"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])

        update_dataframe(df, cur_img, ALGO, resource)

        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER, ALGO)
        
    # OPENCV PROCESSING

        if NEED_WL:
            ALGO = "opencv"

            cur_img.process_ws(ALGO, NEED_WL, resource)
            print("Process "+resource+" of "+ALGO, cur_img.result[resource])

            update_dataframe(df, cur_img, ALGO, resource)

            if NEED_SAVE:
                cur_img.log_labels(LOGS_FOLDER, ALGO)
    
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
            cur_img.log_labels(LOGS_FOLDER, ALGO)
    
    # MAHOTAS PROCESSING
        ALGO = "mahotas"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])

        update_dataframe(df, cur_img, ALGO, resource)

        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER, ALGO)
        
    # ITK PROCESSING
        ALGO = "itk"
        cur_img.process_ws(ALGO, NEED_WL, resource)
        print("Process "+resource+" of "+ALGO, cur_img.result[resource])
        
        
        update_dataframe(df, cur_img, ALGO, resource)

        if NEED_SAVE:
            cur_img.log_labels(LOGS_FOLDER, ALGO)


    if resource == 'time':
        if NEED_WL:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "proc_time_WL.csv"))
        else:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "proc_time.csv"))
    elif resource == 'memory':
        if NEED_WL:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "proc_mem_WL.csv"))
        else:
            df.to_csv(os.path.join(LOGS_FOLDER+"/csv", "proc_mem.csv"))
