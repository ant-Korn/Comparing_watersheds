import sys
import os
import numpy as np
from tools import log_slices
from math import sqrt

sz = sz_x = sz_y = sz_z = 64
rad = 10
filename = 'balls'
indent = 8

if len(sys.argv) > 1:
    sourcedir = sys.argv[1]
else:
    sourcedir = 'source_imgs_3D'

class Sphere:
    def __init__(self, r=1, z=0, x=0, y=0):
        self.r, self.x, self.y, self.z = r, x, y, z

    @property
    def pos(self):
        return (self.z, self.x, self.y)
    
    def is_point_inside(self, z, x, y):
        if (x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2 <= self.r ** 2:
            return True
        return False
    
    def fill(self, mat, value):
        z, x, y = np.ogrid[-self.z:mat.shape[0]-self.z, 
                           -self.x:mat.shape[1]-self.x,
                           -self.y:mat.shape[2]-self.y]
        mask = x*x + y*y + z*z <= self.r*self.r
        mat[mask] = value
    
    def dist(self, sphere):
        dx, dy, dz = self.x-sphere.x, self.y-sphere.y, self.z-sphere.z
        return sqrt(dx*dx+dy*dy+dz*dz)
    
    def fill_gradually(self, mat, min_grey_lvl, max_grey_lvl):
        grey_step = int((max_grey_lvl - min_grey_lvl) / self.r)
        rad_tmp = self.r
        try:
            for grey in range(min_grey_lvl, max_grey_lvl+1, grey_step):
                self.fill(mat, grey)
                self.r -= 1
        finally:
            self.r = rad_tmp



if __name__ == '__main__':
    image = np.zeros((sz_z, sz_x, sz_y), dtype=np.uint8)
    diam = 2 * rad
    start_x = start_y = start_z = indent + rad
    finish_x, finish_y, finish_z = sz_x - indent/2, sz_y - indent/2, sz_z - indent/2
    min_grey, max_grey = 150, 255
    
    x, y, z = start_x, start_y, start_z
    while x + rad < finish_x:
        y = start_y
        while y + rad < finish_y:
            z = start_z
            while z + rad < finish_z:
                Sphere(rad, z, x, y).fill_gradually(image, min_grey, max_grey)
                z += diam + indent
            y += diam + indent
        x += diam + indent
    log_slices(sourcedir, filename, image)
