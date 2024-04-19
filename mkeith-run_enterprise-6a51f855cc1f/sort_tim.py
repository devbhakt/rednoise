#!/usr/bin/env python
from __future__ import print_function

import sys
import numpy as np


pns=[]
lines=[]
with open(sys.argv[1]) as f:
    for line in f:
        e = line.split()
        if len(e) < 7:
            pns.append(-1)
            lines.append(line)
            continue
        for i in range(len(e)):
            if e[i] == '-pn':
                pn=int(e[i+1])
                if pn in pns:
                    pass
                else:
                    pns.append(pn)
                    lines.append(line)
                    break




pns=np.array(pns)


idx = np.argsort(pns)


for i in range(len(idx)):
    line = lines[idx[i]]
    print(line,end='')
