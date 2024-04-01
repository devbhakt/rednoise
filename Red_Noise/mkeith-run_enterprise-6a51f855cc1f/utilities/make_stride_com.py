#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import argparse
import sys, os
import subprocess
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Stride routine for fitting over a glitch. Written by B.Shaw (benjamin.shaw@manchester.ac.uk.')
parser.add_argument('-p', '--parfile', help='Path to ephemeris', required=True)
parser.add_argument('-t', '--timfile', help='Path to tim file', required=True)
parser.add_argument('-w', '--width', help='Boxcar width (days)', required=True)
parser.add_argument('-s', '--stride', help='Size of stride (days)', required=True)
parser.add_argument('-g', '--glep', type=float, default=[], nargs='+', help='Glitch epochs(MJD)')
args = parser.parse_args()
parfile = args.parfile
timfile = args.timfile
width = float(args.width)
stride = float(args.stride)
glep = sorted(args.glep)

def get_lims(tim):
    '''
    Gets first and last MJD from tim file
    '''
    mjds = []
    with open(tim, 'r') as timfile:
        for line in timfile.readlines():
            fields = line.split()
            if len(fields) > 2.0 and fields[0] != "C":
                mjds.append(float(fields[2]))
    #print("Range: {} - {}".format(min(mjds),  max(mjds))) 
    return min(mjds), max(mjds)


def create_global(leading, trailing):
    ''' 
    Set fit timespan using a global par file
    '''
    try:
        os.remove("global.par")
    except OSError:
        pass

    with open("global.par", 'a') as glob:
        for ge in glep:
            if ge is not None:
                if leading < ge and trailing >= ge: 
                    old_trail = trailing
                    trailing = ge - 0.01*stride
                    if trailing - leading < width / 2.0:
                        leading = ge + 0.01*stride
                        trailing = old_trail
        glob.write("START {} 1\n".format(leading))
        glob.write("FINISH {} 1".format(trailing))

    return leading, trailing
 

def main():
    first, last = get_lims(timfile)
    leading = first
    trailing = first + width 
    counter = 0
    while trailing <= last:
        #print("Fitting between {} and {}".format(leading, trailing))
        leading, trailing = create_global(leading, trailing)
        epoch = leading + ((trailing - leading)/2.0)
        print(leading, trailing, epoch)
        
        counter+=1
        leading= first + counter*stride
        trailing= first + width + counter*stride

    

if __name__ == "__main__":
    main()

