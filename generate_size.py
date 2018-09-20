#!/usr/bin/env python
import argparse
from math import sqrt

parser = argparse.ArgumentParser(description='Generate 2D images sizes.')
parser.add_argument("-N", "--number", help="Number of resizes.",
                    type=int)
parser.add_argument("-st", "--step", help="Step of pixels.",
                    type=int)
parser.add_argument("-init", "--init_size", help="Init size of image.",
                    type=int)
args = parser.parse_args()

step = args.step
N = args.number
init_size = args.init_size

print(init_size)

new_size = init_size
pixels = new_size*new_size

for i in range(N):
    pixels += step
    new_size = round(sqrt(pixels))
    print(new_size)
