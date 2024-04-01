import numpy as np
import matplotlib.pyplot as plt
import glob
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument("-t", "--t", type=int, default=[200]*100, nargs='+', help="Replace GLF1 with change of spin frequency 'T' days after the glitch respectively")
parser.add_argument("-fc", "--fc", default="0.02", type=float, help="Corner frequency")
parser.add_argument('-gltd', action='store_true', help="Replace GLF1 with change of spin frequency 'gltd' days after the glitch for all glitches")
arg=parser.parse_args()

file_name_list = glob.glob('*_final.par')
file_name_list.sort()
pulsar_name = [file_name.rsplit('_', 1)[0] for file_name in file_name_list]

for i in range(len(file_name_list)):
    f0=pepoch=redamp=redgam=None
    glitches = {}
    with open(file_name_list[i]) as parfile:
        for line in parfile:
            e = line.strip().split()
            if len(e) > 1:
                if e[0] == "F0":
                    f0 = float(e[1])
                elif e[0] == "PEPOCH":
                    pepoch = float(e[1])
                elif e[0].startswith("GLEP"):
                    glitches[int(e[0][5:])] = {'EP': float(e[1])}
                elif e[0].startswith("GLF0D_"):
                    glitches[int(e[0][6:])]['F0D'] = float(e[1])
                elif e[0].startswith("GLTD_"):
                    glitches[int(e[0][5:])]['TD'] = float(e[1])
                elif e[0].startswith("GLF0D2_"):
                    glitches[int(e[0][7:])]['F0D2'] = float(e[1])
                elif e[0].startswith("GLTD2_"):
                    glitches[int(e[0][6:])]['TD2'] = float(e[1])
                elif e[0].startswith("GLF0D3_"):
                    glitches[int(e[0][7:])]['F0D3'] = float(e[1])
                elif e[0].startswith("GLTD3_"):
                    glitches[int(e[0][6:])]['TD3'] = float(e[1])
                elif e[0].startswith("GLF0_"):
                    glitches[int(e[0][5:])]['F0'] = float(e[1])
                elif e[0].startswith("GLF1_"):
                    glitches[int(e[0][5:])]['F1'] = float(e[1])
                elif e[0].startswith("GLF2_"):
                    glitches[int(e[0][5:])]['F2'] = float(e[1])
                elif e[0] == "TNRedAmp":
                    redamp = float(e[1])
                elif e[0] == "TNRedGam":
                    redgam = float(e[1])
        parfile.close()

    with open('%s_truth.txt'%pulsar_name[i], 'w') as truthfile:
        for gl in glitches:
            glf0=glf1=glf2=glf0_i=glf0_T=None
            glf0d=glf0d2=glf0d3=0
            gltd=gltd2=gltd3=arg.t
            glep = glitches[gl]['EP']
            truthfile.write('GLEP_%i   %f\n'%(gl, glep))
            if 'F0' in glitches[gl]:
                glf0 = glitches[gl]['F0']
                truthfile.write('GLF0_%i   %e\n'%(gl, glf0))
            if 'F1' in glitches[gl]:
                glf1 = glitches[gl]['F1']
                truthfile.write('GLF1_%i   %e\n'%(gl, glf1))
            if 'F2' in glitches[gl]:
                glf2 = glitches[gl]['F2']
                truthfile.write('GLF2_%i   %e\n'%(gl, glf2))
            if 'F0D' in glitches[gl]:
                glf0d = glitches[gl]['F0D']
                truthfile.write('GLF0D_%i   %e\n'%(gl, glf0d))
            if 'TD' in glitches[gl]:
                gltd = glitches[gl]['TD']
                truthfile.write('GLTD_%i   %f\n'%(gl, gltd))
            if 'F0D2' in glitches[gl]:
                glf0d2 = glitches[gl]['F0D2']
                truthfile.write('GLF0D2_%i   %e\n'%(gl, glf0d2))
            if 'TD2' in glitches[gl]:
                gltd2 = glitches[gl]['TD2']
                truthfile.write('GLTD2_%i   %f\n'%(gl, gltd2))
            if 'F0D3' in glitches[gl]:
                glf0d2 = glitches[gl]['F0D3']
                truthfile.write('GLF0D3_%i   %e\n'%(gl, glf0d3))
            if 'TD3' in glitches[gl]:
                gltd3 = glitches[gl]['TD3']
                truthfile.write('GLTD3_%i   %f\n'%(gl, gltd3))
            if all(p is not None for p in [glf0, glf1]):
                if arg.gltd:
                    taug = gltd
                elif gl<=len(arg.t):
                    taug = arg.t[gl-1]
                else:
                    taug = 200
                glf0_i = glf0 + glf0d + glf0d2 + glf0d3
                glf0_T = glf1*taug*86400+glf0d*(np.exp(-taug/gltd)-1)+glf0d2*(np.exp(-taug/gltd2)-1)+glf0d3*(np.exp(-taug/gltd3)-1)
                truthfile.write('GLF0(instant)_%i   %e\n'%(gl, glf0_i))
                truthfile.write('GLF0(T=%d)_%i   %e\n'%(taug, gl, glf0_T))
        if all(p is not None for p in [redamp, redgam]):
            alpha = redgam
            P0 = ((redamp**2)/(12*np.pi**2))*(arg.fc**(-alpha))
            truthfile.write('TNRedAmp   %f\n'%redamp)
            truthfile.write('TNRedGam   %f\n'%redgam)
            truthfile.close()
