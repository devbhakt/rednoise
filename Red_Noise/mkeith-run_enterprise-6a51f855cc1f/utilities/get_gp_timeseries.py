#!/usr/bin/env python
import numpy as np
import scipy.linalg as linalg
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser(description="Run 'enterprise' on a single pulsar")
parser.add_argument('-n','--wavmax',type=int,default=100000)
parser.add_argument('-D','--dm',action='store_true')
parser.add_argument('--subpoly',action='store_true')
parser.add_argument('--f0',type=float,default=None)
parser.add_argument('--pepoch',type=float,default=None)
parser.add_argument('-P','--plot',action='store_true')
parser.add_argument('-s','--start',type=float,default=50000.0)
parser.add_argument('-f','--finish',type=float,default=58000.0)
parser.add_argument('-m','--mjds',type=str)
parser.add_argument('-I','--ifunc',action='store_true')
parser.add_argument('-N','--Nifunc',type=int,default=900)
parser.add_argument('--dmmodel',type=str)

args=parser.parse_args()

lab = np.loadtxt("param.labels",dtype=np.str,usecols=(0,1,2)).T
lab_k = np.loadtxt("param.labels",dtype=np.int,usecols=(2))

beta = np.loadtxt("param.vals")
cvm  = np.loadtxt("cov.matrix")


if args.dm:
    cosidx=lab[1]=='param_red_dm_cos'
    sinidx=lab[1]=='param_red_dm_sin'
    meta = np.loadtxt("tnreddm.meta",usecols=(1))
    dm_epoch=meta[2]
    dm=meta[3]
    dm1=meta[4]
    dm2=meta[5]

else:
    cosidx=lab[1]=='param_red_cos'
    sinidx=lab[1]=='param_red_sin'
    meta = np.loadtxt("tnred.meta",usecols=(1))

omega=meta[0]
epoch=meta[1]

polyidx=lab[1]=='param_f'

wavmax=args.wavmax

cosidx=np.logical_and(cosidx,lab_k<wavmax)
sinidx=np.logical_and(sinidx,lab_k<wavmax)

nwav = np.sum(sinidx)

print("Using {} waves".format(nwav))

if args.ifunc:
    t=np.linspace(args.start,args.finish,args.Nifunc)
else:
    t=np.linspace(args.start,args.finish,10000)

beta_mod = beta[np.logical_or(sinidx,cosidx)]
cvm_mod = cvm[np.logical_or(sinidx,cosidx)][:,np.logical_or(sinidx,cosidx)]

M = np.zeros((2*nwav,len(t)))

for i in range(nwav):
    omegai = omega*(i+1.0)
    M[i]        = np.sin(omegai * (t-epoch))
    M[i+nwav]   = np.cos(omegai * (t-epoch))  


if args.dm:
    tyr=(t-dm_epoch)/365.25
    dmidx=lab[1]=='param_dm'
    npoly=np.sum(dmidx)
    
    M_poly = np.zeros((npoly,len(t)))
    for i in range(npoly):
        M_poly[i] = np.power(tyr,lab_k[dmidx][i])

    print(M.shape,M_poly.shape)
    M = np.concatenate((M_poly,M),axis=0)
    idx = np.logical_or(sinidx,cosidx)
    idx = np.logical_or(dmidx,idx)
    cvm_mod = cvm[idx][:,idx]
    cvm_mod[0]*=0.
    cvm_mod[:,0]*=0.
    beta_mod = beta[idx]

M = M.T


if args.dm:
    tyr=(t-dm_epoch)/365.25
    dm_param=np.logical_and(lab[1]=='param_dm',lab[2]=='0')
    dm_err = np.sqrt(cvm[dm_param,dm_param])
    print("Mean DM error: {}".format(dm_err))
    with open("mean_dm_err.txt","w") as ff:
        ff.write("Mean DM error: {}\n".format(dm_err))
    y = M.dot(beta_mod) + dm1*tyr + dm2*tyr*tyr + dm
    cvm_y = M.dot(cvm_mod).dot(M.T)
    e=np.sqrt(np.diag(cvm_y))
    with open("dm.asc","w") as outf:
        for i in range(len(t)):
            outf.write("{} {} {}\n".format(t[i],y[i],e[i]))




else:
    y = M.dot(beta_mod)
    cvm_y = M.dot(cvm_mod).dot(M.T)
    e=np.sqrt(np.diag(cvm_y))

    model = np.zeros_like(y)

    if args.subpoly:
        f0 = args.f0
        pepoch = args.pepoch
        polybeta = beta[polyidx]
        polycoef=np.array(lab[2,polyidx],dtype=int)
        print(polybeta,polycoef)

        for b,k in zip(polybeta,polycoef):
            if k == 0:
                val = f0 - f0*(1-b/f0)
            else:
                scale = 1
                if k==2:
                    scale= 1e9
                elif k >2 and k<10:
                    scale=1e18
                elif k > 9:
                    scale=1e23

                val = f0 * b/(24*3600)**(k+1)/scale

            print(polycoef,val)
            model = val * (86400.0*(t-pepoch))**(k+1)
            # divide by k factorial
            for i in np.arange(k+1)[1:]:
                model /= i

            #plt.plot(t,model)
            #plt.show()


    if args.ifunc:
        with open("cm.ifunc","w") as outf:
            outf.write("SIFUNC 2 0\n")
            for i in range(len(t)):
                outf.write("IFUNC{} {} {}\n".format(i+1,t[i],-y[i]))
    else:
        with open("cm.asc","w") as outf:
            for i in range(len(t)):
                outf.write("{} {} {}\n".format(t[i],y[i]+model[i],e[i]))







if args.plot:

    plt.figure(figsize=(16,9))
    plt.plot(t,y,color='red')
    plt.fill_between(t,y-e,y+e,color='red',alpha=0.5)

    plt.xlabel("Date")
    if args.dm:
        plt.ylabel("DM")
        plt.errorbar(t[0]-500,y[0],yerr=dm_err,color='red',marker='None',capsize=10)


    if args.dmmodel:
        dmm_t,dmm_d, dmm_e = np.loadtxt(args.dmmodel,usecols=(0,1,2),unpack=True)
        plt.errorbar(dmm_t,dmm_d+dm,yerr=dmm_e,color='k',linestyle=':')

    plt.savefig("plot.pdf")
    plt.show()
