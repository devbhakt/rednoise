#!/usr/bin/env python

from __future__ import print_function
from sys import argv



par=argv[1]
params=argv[2:]


with open(par) as f:
    for line in f:
        e=line.split()
        if len(e) > 0:
            if "-glitch" in params and  e[0].startswith("GL"):
                if "GLF0_" in e[0] or "GLF1_" in e[0] or "GLPH_" in e[0]:
                    print("{:<14} {} 0".format(e[0],e[1]))
                    continue

            if e[0] in params:
                if e[0] == "JUMP":
                    print("{:<14} {} {} {} 0".format(e[0],e[1],e[2],e[3]))
                    continue
                print("{:<14} {}".format(e[0],e[1]))
            else:
                print(line.strip())


