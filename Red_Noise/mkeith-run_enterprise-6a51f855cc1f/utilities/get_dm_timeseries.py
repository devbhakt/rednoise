#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy.linalg as linalg
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser(description="Run 'enterprise' on a single pulsar")
parser.add_argument('-n','--wavmax',type=int,default=100000)
parser.add_argument('-P','--plot',action='store_true')
parser.add_argument('-s','--start',type=float,default=50000.0)
parser.add_argument('-f','--finish',type=float,default=58000.0)
parser.add_argument('-m','--mjds',type=str)
parser.add_argument('-I','--ifunc',action='store_true')
parser.add_argument('-N','--Nifunc',type=int,default=900)
parser.add_argument('--dmmodel-dm',type=float)
parser.add_argument('--dmmodel',type=str)

args=parser.parse_args()

start=args.start
finish=args.finish


lab = np.loadtxt("param.labels",dtype=np.str,usecols=(0,1,2)).T
lab_k = np.loadtxt("param.labels",dtype=np.int,usecols=(2))

beta = np.loadtxt("param.vals")
cvm  = np.loadtxt("cov.matrix")


dmcosidx=lab[1]=='param_red_dm_cos'
dmsinidx=lab[1]=='param_red_dm_sin'
meta = np.loadtxt("tnreddm.meta",usecols=(1))
dm_omega=meta[0]
dm_red_epoch=meta[1]
dm_epoch=meta[2]
dm=meta[3]
dm1=meta[4]
dm2=meta[5]






meta = np.loadtxt("tnred.meta",usecols=(1))
omega=meta[0]
epoch=meta[1]
rx,ry,re,rfreq = np.loadtxt("out.res",usecols=(0,5,6,8),unpack=True)
dat_t=rx
dat_e=re

t=np.linspace(start,finish,200)
y=np.zeros_like(t)

cosidx=lab[1]=='param_red_cos'
sinidx=lab[1]=='param_red_sin'

maxwav=400
nt=0
for i in range(len(cosidx)):
    if cosidx[i]:
        nt+=1
    if nt > maxwav:
        cosidx[i]=False

nt=0
for i in range(len(sinidx)):
    if sinidx[i]:
        nt+=1
    if nt > maxwav:
        sinidx[i]=False

nwav = np.sum(sinidx)
print(nwav,np.sum(dmcosidx))


mask=np.logical_or(np.logical_or(sinidx,cosidx),np.logical_or(dmcosidx,dmsinidx))

dmidx=lab[1]=='param_dm'
npoly=np.sum(dmidx)
mask = np.logical_or(mask,dmidx)
notmask=np.logical_not(mask)

clearmask=lab[1]=='param_ZERO'
#clearmask=np.logical_or(clearmask,lab[1]=='param_f')
print(np.sum(clearmask))

#beta_mod = beta[mask]
#cvm_mod = cvm[mask][:,mask]

dm_param = np.logical_and(dmidx,lab_k==0)
# Zero out the error on DM, we deal with that separately.
dm_err = np.sqrt(cvm[dm_param,dm_param])
cvm[dm_param]*=0.
cvm[:,dm_param]*=0.

print("Disabling parameter",lab.T[np.argwhere(dm_param)])

cvm[np.argwhere(dm_param),np.argwhere(dm_param)]=dm_err*dm_err*1e-20


if False:
    cvm[clearmask]*=0
    cvm[:,clearmask]*=0
print("Mean DM error: {}".format(dm_err))
with open("mean_dm_err.txt","w") as ff:
    ff.write("Mean DM error: {}\n".format(dm_err))
    
print("Total params:",np.sum(mask))

designmatrixfile="design.matrix"

raw_dgnmx = np.genfromtxt(designmatrixfile)

M2 = np.reshape(raw_dgnmx.T[2],(-1,len(beta)))
M2.T[dm_param]*=0.0 # Remove the DM from the design matrix

print(M2.shape)


print("set up matricies")
M = np.zeros((len(beta),len(t)))
dmM = np.zeros((len(beta),len(t)))

kdm=2.41e-4
dmf = np.power(rfreq,-2)/kdm

print(len(dmf),len(dat_t))

ioff=0
tyr=(t-dm_epoch)/365.25
dat_tyr=(dat_t-dm_epoch)/365.25

for iparam in range(len(beta)):
    if cosidx[iparam]:
        omegai = omega*(lab_k[iparam]+1.0)
        M[iparam] = np.cos(omegai * (t-epoch))
    if sinidx[iparam]:
        omegai = omega*(lab_k[iparam]+1.0)
        M[iparam] = np.sin(omegai * (t-epoch))
    if dmcosidx[iparam]:
        omegai = dm_omega*(lab_k[iparam]+1.0)
        dmM[iparam] = np.cos(omegai * (t-dm_red_epoch))
    if dmsinidx[iparam]:
        omegai = dm_omega*(lab_k[iparam]+1.0)
        dmM[iparam] = np.sin(omegai * (t-dm_red_epoch))
    if dmidx[iparam]:
        dmM[iparam] = np.power(tyr,lab_k[iparam])
M=M.T
dmM=dmM.T
        

corr=  np.zeros((len(beta),len(beta)))
for i in range(len(beta)):
    corr[i] = cvm[i]/np.sqrt(cvm[i][i]*np.diag(cvm))

corr -= np.diag(np.zeros(len(beta))+1)

print("Looking for highly covariant parameters:")
for a,b in np.argwhere(corr>0.99):
    print(lab.T[a],lab.T[b],corr[a][b])

for a,b in np.argwhere(corr<-1.99):
    print(lab.T[a],lab.T[b],corr[a][b])
print("Done")

y=M.dot(beta)
y_dat = M2.dot(beta)

W=np.power(dat_e,2)  # np.zeros_like(dat_e)+np.power(0.5e-3,2)#

Coo=M.dot(cvm).dot(M.T)                         
oCf = M2.dot(cvm).dot(M2.T)
Cof = M.dot(cvm).dot(M2.T)        

print("Negative Cf diagonal elements",np.sum(np.diag(oCf)<0))


FF=1.0
while FF < np.power(2,15)+1:
    try:
        print("FF=",FF)
        Cf = oCf + FF*np.diag(W)
        Lf = linalg.cholesky(Cf)
        break
    except Exception as e:
        print(e)
        FF *=2
Lf_inv = linalg.inv(Lf)

#Cf_inv = linalg.inv(Cf)

A=Cof.dot(Lf_inv)
Co = Coo - np.dot(A,A.T)

print("Negative Co: ",np.sum(np.diag(Co)<0))
#print("a^2/d^2 = ",np.power(a/d,2))


#Co *= np.power(b/d,2)
ey = np.sqrt(np.diag(Co))




dmCoo    =dmM.dot(cvm).dot(dmM.T)                         
dmCof    = dmM.dot(cvm).dot(M2.T)        
A=dmCof.dot(Lf_inv)
dmCo = dmCoo - np.dot(A,A.T)
print("Negative dmCo: ",np.sum(np.diag(dmCo)<0))

ydm = dmM.dot(beta) + dm1*tyr + 0.5*dm2*tyr*tyr + dm
edm=np.sqrt(np.diag(dmCo))



with open("dm.asc","w") as outf:
    for i in range(len(t)):
        outf.write("{} {} {}\n".format(t[i],ydm[i],edm[i]))



plt.figure(figsize=(16,9))
plt.plot(t,ydm,color='red')
plt.fill_between(t,ydm-edm,ydm+edm,color='red',alpha=0.5)

plt.xlabel("Date")
plt.ylabel("DM")
plt.errorbar(t[0]-500,[np.mean(ydm)],yerr=dm_err,color='red',marker='None',capsize=10)

if args.dmmodel:
    dmmodel_dm=dm
    if not (args.dmmodel_dm is None):
        dmmodel_dm = args.dmmodel_dm
    dmm_t,dmm_d, dmm_e = np.loadtxt(args.dmmodel,usecols=(0,1,2),unpack=True)
    plt.errorbar(dmm_t,dmm_d+dmmodel_dm,yerr=dmm_e,color='k',linestyle=':')

plt.savefig("plot.pdf")
if args.plot:
    plt.show()
