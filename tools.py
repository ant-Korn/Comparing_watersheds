import os
import numpy as np
from scipy import ndimage
import pandas as pd
from skimage.feature import peak_local_max
from memory_profiler import memory_usage
import time
import math

from skimage import filters, morphology, transform
import cv2
from smilPython import Image, watershed, CubeSE, SquSE, basins
from mamba import imageMb, SQUARE, copyBytePlane, watershedSegment, basinSegment
from mamba3D import image3DMb, CUBIC, copyBytePlane3D, watershedSegment3D, basinSegment3D
from mahotas import cwatershed
import itk


def get_spaced_colors(n):
    if n < 10:
        n = 10
    max_value = 256**3
    interval = max_value // n
    colors = [hex(int_value)[2:].zfill(6) for int_value in range(0, max_value, interval)]
    palette = np.array([(int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)) for color in colors])
    np.random.shuffle(palette[1:])
    return palette[:n]

def log_slices_2d(logfoldername, logname, i2d):
    fname = os.path.join(logfoldername, logname + ".png")
    cv2.imwrite(fname, i2d)

def log_slices(logfoldername, logname, i3d):
    fname = os.path.join(logfoldername, logname + "{0:04}.png")
    for i in range(0, i3d.shape[0]):
        cv2.imwrite(fname.format(i + 1), i3d[i])

def log_labels_2d(logfoldername, logname, i2d, flags=None):
    fname = os.path.join(logfoldername, logname + ".png")
    nlabels = np.max(i2d) + 1

    palette = get_spaced_colors(nlabels)
    if flags is not None:
        palette[:, 0] = palette[:, 0] * flags
        palette[:, 1] = palette[:, 1] * flags
        palette[:, 2] = palette[:, 2] * flags

    rgb_slice = np.zeros((i2d.shape[0], i2d.shape[1], 3), dtype=np.uint8)
    rgb_slice = palette[i2d]
    cv2.imwrite(fname, rgb_slice)

def log_labels(logfoldername, logname, i3d, flags=None):
    fname = os.path.join(logfoldername, logname + "{0:04}.png")
    nlabels = np.max(i3d) + 1

    palette = get_spaced_colors(nlabels)
    if flags is not None:
        palette[:, 0] = palette[:, 0] * flags
        palette[:, 1] = palette[:, 1] * flags
        palette[:, 2] = palette[:, 2] * flags

    rgb_slice = np.zeros((i3d.shape[1], i3d.shape[2], 3), dtype=np.uint8)
    for i in range(0, i3d.shape[0]):
        rgb_slice = palette[i3d[i]]
        cv2.imwrite(fname.format(i + 1), rgb_slice)

def read_slices_2d(input_name):
    return cv2.imread(input_name, cv2.IMREAD_GRAYSCALE)

def read_slices(input_name, num_slices, start_index=1):
    if num_slices < 0:
        return None
    islice = cv2.imread(input_name.format(start_index), cv2.IMREAD_GRAYSCALE)
    i3d = np.zeros((num_slices,) + islice.shape, dtype=islice.dtype)
    i3d[0] = islice
    for i in range(1, num_slices):
        islice = cv2.imread(input_name.format(
            i + start_index), cv2.IMREAD_GRAYSCALE)
        i3d[i] = islice
    return i3d

def getArrayFromImage(imIn, size):
    """
    Creates an 2D array containing the same data as in 'imIn'. Only
    works for greyscale and 32-bit images. Returns the array.
    """
    if imIn.getDepth()==8:
        dtype = np.uint8
    elif imIn.getDepth()==32:
        dtype = np.uint32
    else:
        raiseExceptionOnError()
            
    (w,h) = imIn.getSize()
    # First extracting the raw data out of image imIn
    data = imIn.extractRaw()
    # creating an array with this data
    # At this step this is a one-dimensional array
    array1D = np.fromstring(data, dtype=dtype)
    # Reshaping it to the dimension of the image
    array2D = array1D.reshape((h,w))
    array2D = np.array(array2D[0:size, 0:size])
    return array2D

def getArrayFromImage_3D(imIn, size):
    """
    Creates an 3D array containing the same data as in 'imIn'. Only
    works for greyscale and 32-bit images. Returns the array.
    """
    if imIn.getDepth()==8:
        dtype = np.uint8
    elif imIn.getDepth()==32:
        dtype = np.uint32
    else:
        raiseExceptionOnError()
            
    (w,h,d) = imIn.getSize()
    # First extracting the raw data out of image imIn
    data = imIn.extractRaw()
    # creating an array with this data
    # At this step this is a one-dimensional array
    array1D = np.fromstring(data, dtype=dtype)
    # Reshaping it to the dimension of the image
    array3D = array1D.reshape((h,w,d))
    array3D = np.array(array3D[0:size, 0:size, 0:size])
    return array3D

def fillImageWithArray(array, imOut):
    """
    Fills image 'imOut' with the content of two dimensional 'array'. Only
    works for greyscale and 32-bit images.
    """
    # Checking depth 
    if (imOut.getDepth()==8 and array.dtype != np.uint8) or \
            (imOut.getDepth()==32 and array.dtype != np.uint32) or \
            (imOut.getDepth()==1):
        raiseExceptionOnError()

    # image size
    (wi,hi) = imOut.getSize()
    # array size
    (ha,wa) = array.shape

    # Checking the sizes
    #if wa!=wi or ha!=hi:
    #    raiseExceptionOnError()
    
    pad_w = wi - wa
    pad_h = hi - ha
    data = np.pad(array, ((0, pad_h), (0, pad_w)) , mode='constant')
    data = data.tostring()
    imOut.loadRaw(data)

def fillImageWithArray_3D(array, imOut):
    """
    Fills image 'imOut' with the content of three dimensional 'array'. Only
    works for greyscale and 32-bit images.
    """
    # Checking depth 
    if (imOut.getDepth()==8 and array.dtype != np.uint8) or \
            (imOut.getDepth()==32 and array.dtype != np.uint32) or \
            (imOut.getDepth()==1):
        raiseExceptionOnError()

    # image size
    (wi,hi,di) = imOut.getSize()
    # array size
    (ha,wa,da) = array.shape

    # Checking the sizes
    #if wa!=wi or ha!=hi:
    #    raiseExceptionOnError()
    
    pad_w = wi - wa
    pad_h = hi - ha
    pad_d = di - da
    data = np.pad(array, ((0, pad_h), (0, pad_w), (0, pad_d)) , mode='constant')
    data = data.tostring()
    imOut.loadRaw(data)

def label_2d_cv_2(img, min_dist, need_invert=False):
    thresh = cv2.threshold(img, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if need_invert:
        thresh = 255 - thresh
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=min_dist,
            labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    return (thresh, D, markers)

def create_itk_img(sz, dim, typeimg):
    image1 = itk.Image[typeimg, dim].New()
    size = itk.Size[dim]()
    size = sz
    region = itk.ImageRegion[dim]()
    region.SetSize(size)
    image1.SetRegions(region)
    image1.Allocate()
    return image1

def create_data_frame(imgs, iterables):
    dictionary = {k: {k1: v1 
                           for (k1,v1) in zip(iterables[1], 
                                              [list(map(lambda x: x.split('.',1)[0].rsplit('_',1)[0],
                                                   imgs
                                                  )),
                                               list(map(lambda x: int(x.split('.',1)[0].rsplit('_',1)[1]),
                                                   imgs
                                                  )),
                                               [0.0 for i in range(len(imgs))]
                                              ]
                                             ) 
                          }
                       for k in iterables[0] }
    dictionary = {(outerKey, innerKey): values for outerKey, innerDict in dictionary.items() for innerKey, values in innerDict.items()}

    return pd.DataFrame(dictionary)

def create_data_frame_3D(iterables, filename, sizes_list):
    dictionary = {k: {k1: v1 
                           for (k1,v1) in zip(iterables[1], 
                                              [[filename for i in range(len(sizes_list))],
                                               sizes_list,
                                               [0.0 for i in range(len(sizes_list))]
                                              ]
                                             ) 
                          }
                       for k in iterables[0] }
    dictionary = {(outerKey, innerKey): values for outerKey, innerDict in dictionary.items() for innerKey, values in innerDict.items()}
    return pd.DataFrame(dictionary)

def update_dataframe(df, img, ALGO, resource):
    res = img.result[resource]
    df.loc[(df[ALGO]['img_name']==img.cluster) & (df[ALGO]['size']==img.size), (ALGO, resource)] = res

def decorator(funk_ws, res_type):
    def return_funk(*args, **kwargs):
        if res_type == 'time':
            start_t = time.process_time()
            labels = funk_ws(*args, **kwargs)
            process_time = (time.process_time() - start_t) 
            return (labels, process_time)
        elif res_type == 'memory':
            labels = None
            tmp = memory_usage((funk_ws, args, kwargs), interval=0.01)
            max_mem = np.max(tmp)
            return (labels, max_mem)
    return return_funk

class Image_2D:
    def __init__(self, folder, filename, init_size):
        self.folder, self.filename = folder, filename
        self.init_size = init_size
        self.size = int(filename.split('.', 1)[0].rsplit('_', 1)[1])
        self.cluster = filename.split('.', 1)[0].rsplit('_', 1)[0] 
        self.mul_size = self.size / init_size
        self.img = cv2.imread(os.path.join(folder, filename))
        self.result = {'time': 0.0, 'memory': 0.0}

    def greyscale(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def init_label(self, INIT_MIN_DIST, image_need_invert):
        self.bw, self.dist, self.markers = label_2d_cv_2(self.greyscale(),
                                                         int(INIT_MIN_DIST*self.mul_size),
                                                         (self.cluster in image_need_invert))
        #Converting dist to inverse dist
        self.dist = int(np.max(self.dist)) * np.ones(self.dist.shape, np.uint32) - self.dist.astype(np.uint32)
        self.labels = np.zeros(self.dist.shape, dtype=self.markers.dtype)

    def log_labels(self, LOGS_FOLDER, ALGO):
        self.labels *= self.bw
        log_labels_2d(LOGS_FOLDER, ALGO + "_" + self.filename.split('.', 1)[0], self.labels)

    def process_ws(self, ALGO, NEED_WL, res_type):
        if ALGO == 'skimage':
            self.result[res_type] = self.skimage_ws(NEED_WL, res_type)
        elif ALGO == 'opencv':
            self.result[res_type] = self.opencv_ws(res_type)
        elif ALGO == 'smil':
            self.result[res_type] = self.smil_ws(NEED_WL, res_type)
        elif ALGO == 'mamba':
            self.result[res_type] = self.mamba_ws(NEED_WL, res_type)
        elif ALGO == 'mahotas':
            self.result[res_type] = self.mahotas_ws(NEED_WL, res_type)
        elif ALGO == 'itk':
            self.result[res_type] = self.itk_ws(NEED_WL, res_type)

    def _skimage_ws(self, NEED_WL):
        return morphology.watershed(self.dist, self.markers,
                                    connectivity=ndimage.generate_binary_structure(2, 2),
                                    watershed_line=NEED_WL)

    def skimage_ws(self, NEED_WL, res_type):
        self.labels, res = decorator(Image_2D._skimage_ws, res_type)(self, NEED_WL)
        return res

    def _opencv_ws(self):
        return cv2.watershed(self.img, self.markers)

    def opencv_ws(self, res_type):
        tmp_mar = np.zeros(self.markers.shape, self.markers.dtype)
        tmp_mar[:] = self.markers[:]
        self.markers, res = decorator(Image_2D._opencv_ws, res_type)(self)
        self.labels[:] = self.markers[:]
        self.markers[:] = tmp_mar[:]
        return res

    def smil_ws(self, NEED_WL, res_type):
        # Load an image
        imIn = Image(os.path.join(self.folder, self.filename))

        # Create a gradient image
        imGrad = Image(imIn, "UINT16")
        imArr = imGrad.getNumArray()
        imArr[:] =  np.transpose(self.dist)

        # Manually impose markers on image
        imMark = Image(imIn, "UINT16")
        imArr = imMark.getNumArray() 
        imArr[:] = np.transpose(self.markers)

        # Create the watershed
        imBS = Image(imIn, "UINT16")

        if NEED_WL:
            imWS = Image(imIn, "UINT16")
            _, res = decorator(watershed, res_type)(imGrad, imMark, imWS, imBS, SquSE())
        else:
            _, res = decorator(basins, res_type)(imGrad, imMark, imBS, SquSE())


        imArr = imBS.getNumArray()
        self.labels[:] = np.transpose(imArr)
        return res

    def mamba_ws(self, NEED_WL, res_type):
        imDist = imageMb(*self.dist.shape, 32)
        fillImageWithArray(self.dist, imDist)
        imMarkers = imageMb(*self.markers.shape, 32)
        fillImageWithArray(self.markers.astype(np.uint32), imMarkers)

        if NEED_WL:
            labels, res = decorator(watershedSegment, res_type)(imDist, imMarkers, grid=SQUARE)
        else:
            labels, res = decorator(basinSegment,res_type)(imDist, imMarkers, grid=SQUARE)

        tmp = imageMb(*self.markers.shape, 8)
        copyBytePlane(tmp, 3, imMarkers)
        self.labels = getArrayFromImage(imMarkers, self.size)
        return res

    def _mahotas_ws(self, NEED_WL):
        if NEED_WL:
            labels, wl = cwatershed(self.dist, self.markers, Bc=ndimage.generate_binary_structure(2, 2), return_lines=NEED_WL)
        else: 
            labels = cwatershed(self.dist, self.markers, Bc=ndimage.generate_binary_structure(2, 2), return_lines=NEED_WL)
        return labels

    def mahotas_ws(self, NEED_WL, res_type):
        self.labels, res = decorator(Image_2D._mahotas_ws, res_type)(self, NEED_WL)
        return res

    def _itk_ws(self, itk_filter):
        itk_filter.Update() 
        labelImage = itk_filter.GetOutput()
        return labelImage

    def itk_ws(self, NEED_WL, res_type):
        Dimension = 2

        dist_itk = create_itk_img(self.dist.shape, Dimension, itk.SS)
        markers_itk = create_itk_img(self.dist.shape, Dimension, itk.SS)

        np_view = itk.GetArrayViewFromImage(dist_itk)
        np_mar = itk.GetArrayViewFromImage(markers_itk)
        np_view[:] = self.dist
        np_mar[:] = self.markers
        f = itk.MorphologicalWatershedFromMarkersImageFilter.New()
        f.SetMarkWatershedLine(NEED_WL)
        f.SetFullyConnected(True)
        f.SetInput1(dist_itk)
        f.SetInput2(markers_itk)
        labelImage, res = decorator(Image_2D._itk_ws, res_type)(self, f)
        if labelImage is not None:
            self.labels[:] = itk.GetArrayViewFromImage(labelImage)
        return res


def gen_sizes_3D(N, step_pix, init_size):
    sizes = []
    pixels = math.pow(init_size, 3)
    for i in range(N):
        sizes.append(round(math.pow(pixels, 1.0/3.0)))
        pixels += step_pix
    return sizes

def image_generator(IMGS_FOLDER, filename, INIT_SIZE, sizes):
    init_img = Image_3D(IMGS_FOLDER, filename, INIT_SIZE)
    for size in sizes:
        if size == INIT_SIZE:
            yield init_img
        else:
            yield Image_3D(IMGS_FOLDER, filename, size, init_img)

class Image_3D:
    def __init__(self, folder, filename, size, init_img=None):
        self.cluster = filename
        self.size = size
        if init_img is None:
            path_initial_image = os.path.join(folder, filename+'{0:04}.png')
            self.img = read_slices(path_initial_image, size)
            self.init_shape = (size, size, size)
        else:
            self.img = self.resize_img(init_img.img, size)
            self.init_shape = init_img.img.shape
        self.result = {'time': 0.0, 'memory': 0.0}

    def resize_img(self, img, size):
        return transform.resize(img, (size, size, size), order=0, preserve_range=True, mode='edge', anti_aliasing=False).astype(int)
        
    def init_label(self, rad, indent):
        self.bw = (self.img != 0)
        self.dist = ndimage.distance_transform_edt(self.bw)
        self.markers = np.zeros(self.init_shape, dtype=np.int32)
        diam = 2 * rad
        start_x = start_y = start_z = indent + rad
        sz_x, sz_y, sz_z = self.init_shape
        finish_x, finish_y, finish_z = sz_x - indent/2, sz_y - indent/2, sz_z - indent/2
        
        mark = 1
        x, y, z = start_x, start_y, start_z
        while x + rad < finish_x:
            y = start_y
            while y + rad < finish_y:
                z = start_z
                while z + rad < finish_z:
                    self.markers[x][y][z] = mark
                    mark += 1
                    z += diam + indent
                y += diam + indent
            x += diam + indent
        self.markers = self.resize_img(self.markers, self.size)
        #Converting dist to inverse dist
        self.dist = int(np.max(self.dist)) * np.ones(self.dist.shape, np.uint32) - self.dist.astype(np.uint32)

    def log_labels(self, LOGS_FOLDER, ALGO):
        self.labels *= self.bw
        log_labels(LOGS_FOLDER, ALGO+'_'+self.filename, self.labels)

    def process_ws(self, ALGO, NEED_WL, res_type):
        if ALGO == 'skimage':
            self.result[res_type] = self.skimage_ws(NEED_WL, res_type)
        elif ALGO == 'smil':
            self.result[res_type] = self.smil_ws(NEED_WL, res_type)
        elif ALGO == 'mamba':
            self.result[res_type] = self.mamba_ws(NEED_WL, res_type)
        elif ALGO == 'mahotas':
            self.result[res_type] = self.mahotas_ws(NEED_WL, res_type)
        elif ALGO == 'itk':
            self.result[res_type] = self.itk_ws(NEED_WL, res_type)

    def _skimage_ws(self, NEED_WL):
        return morphology.watershed(self.dist, self.markers,
                                    connectivity=ndimage.generate_binary_structure(3, 3),
                                    watershed_line=NEED_WL)

    def skimage_ws(self, NEED_WL, res_type):
        self.labels, res = decorator(Image_3D._skimage_ws, res_type)(self, NEED_WL)
        return res

    def smil_ws(self, NEED_WL, res_type):
        # Load an image
        imIn = Image(*self.img.shape)
        imArr = imIn.getNumArray()
        imArr[:] = self.img

        # Create a gradient image
        imGrad = Image(imIn, "UINT16")
        imArr = imGrad.getNumArray()
        imArr[:] =  np.transpose(self.dist)

        # Manually impose markers on image
        imMark = Image(imIn, "UINT16")
        imArr = imMark.getNumArray() 
        imArr[:] = np.transpose(self.markers)

        # Create the watershed
        imBS = Image(imIn, "UINT16")

        if NEED_WL:
            imWS = Image(imIn, "UINT16")
            _, res = decorator(watershed, res_type)(imGrad, imMark, imWS, imBS, CubeSE())
        else:
            _, res = decorator(basins, res_type)(imGrad, imMark, imBS, CubeSE())


        imArr = imBS.getNumArray()
        self.labels[:] = np.transpose(imArr)
        return res

    def mamba_ws(self, NEED_WL, res_type):
        imDist = image3DMb(*self.dist.shape, 32)
        fillImageWithArray_3D(self.dist, imDist)
        imMarkers = image3DMb(*self.markers.shape, 32)
        fillImageWithArray_3D(self.markers.astype(np.uint32), imMarkers)

        if NEED_WL:
            labels, res = decorator(watershedSegment3D, res_type)(imDist, imMarkers, grid=CUBIC)
        else:
            labels, res = decorator(basinSegment3D,res_type)(imDist, imMarkers, grid=CUBIC)

        tmp = image3DMb(*self.markers.shape, 8)
        copyBytePlane3D(tmp, 3, imMarkers)
        self.labels = getArrayFromImage_3D(imMarkers, self.size)
        return res

    def _mahotas_ws(self, NEED_WL):
        if NEED_WL:
            labels, wl = cwatershed(self.dist, self.markers, Bc=ndimage.generate_binary_structure(3, 3), return_lines=NEED_WL)
        else: 
            labels = cwatershed(self.dist, self.markers, Bc=ndimage.generate_binary_structure(3, 3), return_lines=NEED_WL)
        return labels

    def mahotas_ws(self, NEED_WL, res_type):
        self.labels, res = decorator(Image_3D._mahotas_ws, res_type)(self, NEED_WL)
        return res

    def _itk_ws(self, itk_filter):
        itk_filter.Update() 
        labelImage = itk_filter.GetOutput()
        return labelImage

    def itk_ws(self, NEED_WL, res_type):
        Dimension = 3

        dist_itk = create_itk_img(self.dist.shape, Dimension, itk.SS)
        markers_itk = create_itk_img(self.dist.shape, Dimension, itk.SS)

        np_view = itk.GetArrayViewFromImage(dist_itk)
        np_mar = itk.GetArrayViewFromImage(markers_itk)
        np_view[:] = self.dist
        np_mar[:] = self.markers
        f = itk.MorphologicalWatershedFromMarkersImageFilter.ISS3ISS3.New()
        f.SetMarkWatershedLine(NEED_WL)
        f.SetFullyConnected(True)
        f.SetInput1(dist_itk)
        f.SetInput2(markers_itk)
        labelImage, res = decorator(Image_3D._itk_ws, res_type)(self, f)
        if labelImage is not None:
            self.labels[:] = itk.GetArrayViewFromImage(labelImage)
        return res

