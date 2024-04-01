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
puls = str(timfile).split('.')[0]
pname = puls.split('_')[-1]
print(pname)


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


def run_fit(par, tim, epoch):
    '''
    Run tempo2 and fit for parameters
    '''
    epoch = str(epoch)
    command = [
              'tempo2', '-f', par, tim, 
              '-nofit', '-global', 'global.par',  
              '-fit', 'F0', '-fit', 'F1', '-fit', 'F2',
              #'-fit', 'F0', '-fit', 'F1',
              '-epoch', epoch
              ] 
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    #print(proc.stdout.read)
    while proc.poll() is None:
        l = proc.stdout.readline().decode('utf-8')
        fields = l.split()
        if len(fields) > 4:
            if fields[0] == "PEPOCH":
                pepoch = fields[3]
            if fields[0] == "F0":
                F0 = fields[3]
                F0_e = fields[4]
                if not 0<abs(float(F0_e))<1e-7: 
                    return None
                #if float(F0_e)==0 or F0_e=='nan' or F0_e=='-nan': 
                    #return None
            if fields[0] == "F1":
                F1 = fields[3]
                F1_e = fields[4]
                if not 0<abs(float(F1_e))<1e-14: 
                    return None
                #if float(F1_e)==0 or F1_e=='nan' or F1_e=='-nan': 
                    #return None
            if fields[0] == "F2":
                F2 = fields[3]
                F2_e = fields[4]
                if not 0<abs(float(F2_e))<1e-17: 
                    return None
                #if float(F2_e)==0 or F2_e=='nan' or F2_e=='-nan': 
                    #return None
    try:    
        return pepoch, F0, F0_e, F1, F1_e, F2, F2_e
    except UnboundLocalError:
        #print("here")
        return None


def main():
    first, last = get_lims(timfile)
    leading = first
    trailing = first + width 
    counter = 0
    epochs = str(pname)+'_g'+str(len(glep))+'_w'+str(int(width))+'_s'+str(int(stride))+'_epoch.txt'
    with open(epochs, 'w') as f1:
        while trailing <= last:
            #print("Fitting between {} and {}".format(leading, trailing))
            leading, trailing = create_global(leading, trailing)
            epoch = leading + ((trailing - leading)/2.0)
            print(leading, trailing, epoch, file=f1)
        
            counter+=1
            leading= first + counter*stride
            trailing= first + width + counter*stride

    starts, ends, fitepochs = np.loadtxt(epochs, unpack=True)
    with open(str(pname)+'_g'+str(len(glep))+'_w'+str(int(width))+'_s'+str(int(stride))+'_data.txt', 'w') as f2:
        for i in range(0, len(starts)):
            #print("Fitting between {} and {}".format(starts[i], ends[i]))
            create_global(starts[i], ends[i])
            epoch = fitepochs[i]
        
            out = run_fit(parfile, timfile, epoch)
            os.remove("global.par")
            #os.rename("global.par", str(starts[i]) + "_" + str(ends[i]) + ".eph")
            if out:
                print(out[0], out[1], out[2], out[3], out[4], out[5], out[6], starts[i], ends[i], file=f2)


if __name__ == "__main__":
    main()




