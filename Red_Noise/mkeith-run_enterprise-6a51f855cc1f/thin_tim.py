#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np

gl=[]

delta=float(sys.argv[3])

with open(sys.argv[1]) as f:
    for line in f:
        e=line.split()
        if e[0].startswith("GLEP"):
            gl.append(float(e[1]))


gl=np.array(gl)


count_in=0
count_out=0
last=0
pns=[]
lines=[]
with open(sys.argv[2]) as f:
    for line in f:
        e = line.split()
        if len(e) < 5:
            print(line,end='')
            continue

        count_in+=1

        mjd = float(e[2])
        if mjd-last > delta or np.amin(np.abs(mjd-gl)) < 100:
            print(line,end='')
            last=mjd
            count_out+=1






print("# IN {}   OUT {}".format(count_in,count_out),file=sys.stderr)
