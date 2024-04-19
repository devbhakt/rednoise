#!/usr/bin/env python
from __future__ import print_function
# Import the libraries we need.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import linalg
import sys
import subprocess

par=sys.argv[1]
tim=sys.argv[2]
par2=sys.argv[3]

subprocess.call(["tempo2","-output","exportres","-f",par2,tim,"-nofit"])

rx2,ry2,re2 = np.loadtxt("out.res",usecols=(0,5,6),unpack=True)

subprocess.call(["tempo2","-output","exportres","-f",par,tim,"-writeres"])
#subprocess.run(["ls","-l"])

lab = np.loadtxt("param.labels",dtype=np.str).T
beta = np.loadtxt("param.vals")
cvm  = np.loadtxt("cov.matrix")
meta = np.loadtxt("tnred.meta",usecols=(1))
F0=0
start =46500
finish=58200
psrn="??"

glep=np.zeros(100)
glf0=np.zeros(100)
glf1=np.zeros(100)

glf0d=np.zeros(100)
gltd=np.zeros(100)

glf0d2=np.zeros(100)
gltd2=np.zeros(100)
max_glitch=0
PB=0

inpar=[]

with open(par) as f:
    for line in f:
        if not line.startswith("TNRed"):
            inpar.append(line)
        line = line.strip()
        e=line.split()
        if e[0] == "PSRJ":
            psrn=e[1]
        if e[0].startswith("GLEP_"):
            i=int(e[0][5:])
            glep[i-1] = float(e[1])
            max_glitch = max(i,max_glitch)
        if e[0].startswith("GLF0_"):
            i=int(e[0][5:])
            glf0[i-1] = float(e[1])
        if e[0].startswith("GLTD_"):
            i=int(e[0][5:])
            gltd[i-1] = float(e[1])
        if e[0].startswith("GLF0D_"):
            i=int(e[0][6:])
            glf0d[i-1] = float(e[1])
        if e[0].startswith("GLTD2_"):
            i=int(e[0][6:])
            gltd2[i-1] = float(e[1])
        if e[0].startswith("GLF0D2_"):
            i=int(e[0][7:])
            glf0d2[i-1] = float(e[1])

        if e[0].startswith("GLF1_"):
            i=int(e[0][5:])
            glf1[i-1] = float(e[1])
        if e[0] == "F0":
            F0=float(e[1])
        if e[0] == "PB":
            PB=float(e[1])
        if e[0] == "F1":
            F1=float(e[1])
        if e[0] == "F2":
            F2=float(e[1])
        if e[0] == "START":
            start=float(e[1])
        if e[0] == "FINISH":
            finish=float(e[1])
        if e[0] == "PEPOCH":
            pepoch=float(e[1])
            
glep=glep[:max_glitch]
omega=meta[0]
epoch=meta[1]

rx,ry,re = np.loadtxt("out.res",usecols=(0,5,6),unpack=True)

dat_t=rx
dat_e=re
NSAMPLES=1000

t=np.linspace(start-0.5,finish+0.5,NSAMPLES)
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

beta_mod = beta[np.logical_or(sinidx,cosidx)]
cvm_mod = cvm[np.logical_or(sinidx,cosidx)][:,np.logical_or(sinidx,cosidx)]

cvm_mod_orig = np.copy(cvm_mod)

frqidx=lab[1]=='param_f'

CCof = cvm[np.logical_or(sinidx,cosidx),:][:,frqidx]
CCf  = cvm[frqidx,:][:,frqidx]
LLf = linalg.cholesky(CCf)
LLfinv = linalg.inv(LLf)

A=CCof.dot(LLfinv)
cvm_mod_unf = cvm_mod_orig - np.dot(A,A.T)

cvm_mod=np.diag(np.diag(cvm_mod_unf))


Lcvm = linalg.cholesky(cvm_mod)

s2d2 = 1.0/(86400.0*86400.0)
s2d = 1.0/86400.0

M = np.zeros((2*nwav,len(y)))
M2 = np.zeros((2*nwav,len(dat_t)))

dM = np.zeros_like(M)
ddM = np.zeros_like(M)

with open("white.par","w") as f:
    f.writelines(inpar)
    f.write("WAVE_OM {}\n".format(omega))
    f.write("WAVEEPOCH {}\n".format(epoch))
    for i in range(min(256,nwav)):
        f.write("WAVE{}  {}  {}\n".format(i+1,-beta_mod[i],-beta_mod[i+nwav]))




print("set up matricies")

freqs=[]
pwrs=np.power(beta_mod[:nwav],2) + np.power(beta_mod[nwav:],2)

for i in range(nwav):
    omegai = omega*(i+1.0)
    M[i]        = np.sin(omegai * (t-epoch))
    M[i+nwav]   = np.cos(omegai * (t-epoch))

    freqs.append(365.25*omegai/2.0/np.pi)
    
    dM[i]      = -F0*omegai*s2d*M[i+nwav]
    dM[i+nwav] = F0*omegai*s2d*M[i]
    
    ddM[i]      = 1e15*F0*omegai*omegai*s2d2*M[i]
    ddM[i+nwav] = 1e15*F0*omegai*omegai*s2d2*M[i+nwav]
                         
    M2[i]       = np.sin(omegai * (dat_t-epoch))
    M2[i+nwav]  = np.cos(omegai * (dat_t-epoch))
  
print("Do linear algebra")

freqs=np.array(freqs)


maxP=2*np.pi/omegai
                         
                         
#print(cvm.shape,cvm_mod.shape)
M = M.T
ddM = ddM.T
M2 = M2.T
dM = dM.T

y=M.dot(beta_mod)

y_dat = M2.dot(beta_mod)




"""
    if (t < mjd[0]){
        // we are before the first jump
        // so our gradient is just the zeroth offset.
        return yoffs[0];
    } else if(t > mjd[N-1]){
        return yoffs[N-1];
    } else{
        // find the pair we are between...
        for (int ioff =0;ioff<N;ioff++){
            if(t >= mjd[ioff] && t < mjd[ioff+1]){
                double x1 = mjd[ioff];
                double x2 = mjd[ioff+1];
                double x = (t-x1)/(x2-x1);
                double y1=yoffs[ioff];
                double y2=yoffs[ioff+1];
                return (y2-y1)*x + y1;
            }
        }
    }
"""

def ifunc(t,mjd,yoffs):
    if t < mjd[0]:
        return yoffs[0]
    elif t > mjd[-1]:
        return yoffs[-1]
    else:
        for i in range(len(mjd)):
            if t > mjd[i] and t < mjd[i+1]:
                x1=mjd[i]
                x2=mjd[i+1]
                x=(t-x1)/(x2-x1)
                y1=yoffs[i]
                y2=yoffs[i+1]
                return (y2-y1)*x + y1


W=np.power(dat_e,2)  # np.zeros_like(dat_e)+np.power(0.5e-3,2)#
Coo=M.dot(cvm_mod).dot(M.T)                         
Cf = M2.dot(cvm_mod).dot(M2.T) 
Cof = M.dot(cvm_mod).dot(M2.T)        

#Cf = correlation_tools.cov_nearest(Cf,n_fact=1000,method="clipped") 

#print(linalg.eigvals(Cf))
#origCf = np.copy(Cf)
#p_m = np.trace(Cf)/float(len(dat_e))
#d=10000000.0
#b=0.0
#a=d
#while True:
#    try:
#        #print(numpy.linalg.cond(Cf,'fro'))
#        print("T",a)
#        Lf = linalg.cholesky(Cf)
#        
#        break
#    except:
#        a-=2
#        b = np.sqrt(d*d-a*a)
#        Cf = np.power(a/d,2)*origCf + np.power(b/d,2) * p_m*np.eye(len(dat_e))
#        #print(np.power(a/d,2),p_m,Cf[0][0],origCf[0][0])
#        #print(origCf[0][0],np.power(a/d,2)*origCf[0][0],np.power(b/d,2) * p_m)
#print(a)

Cf+= 5*np.diag(W)
Lf = linalg.cholesky(Cf)
Lf_inv = linalg.inv(Lf)

Cf_inv = linalg.inv(Cf)
#Co = Coo - np.dot(np.power(a/d,2)*Cof,np.dot(Cf_inv,np.power(a/d,2)*Cof.T))
#Co = Coo - np.dot(Cof,np.dot(Cf_inv,Cof.T))

A=Cof.dot(Lf_inv)
Co = Coo - np.dot(A,A.T)

print("Negative Co: ",np.sum(np.diag(Co)<0))
#print("a^2/d^2 = ",np.power(a/d,2))


#Co *= np.power(b/d,2)
ey = np.sqrt(np.diag(Co))





ddCoo    =ddM.dot(cvm_mod).dot(ddM.T)                         
ddCof    = ddM.dot(cvm_mod).dot(M2.T)        
A=ddCof.dot(Lf_inv)
ddCo = ddCoo - np.dot(A,A.T)
print("Negative ddCo: ",np.sum(np.diag(ddCo)<0))

yd = ddM.dot(beta_mod)
ed=np.sqrt(np.diag(ddCo))

tt=(t-pepoch)*86400.0
yd_model=np.zeros_like(yd)
yd_model += 1e15*(F2*tt + F1)

i=0
for ge in glep:
    print("Glitch {} = {} >> {}".format(i,ge,1e15*glf1[i]))
    yd_model[t>ge] += 1e15*glf1[i]
    if gltd[i] > 0:
        print("Exponential")
        yd_model[t>ge] -= 1e15 * glf0d[i] * np.exp(-(t[t>ge]-glep[i])/gltd[i]) / (gltd[i]*86400.0)
    if gltd2[i] > 0:
        print("2xExponential")
        yd_model[t>ge] -= 1e15 * glf0d2[i] * np.exp(-(t[t>ge]-glep[i])/gltd2[i]) / (gltd2[i]*86400.0)

    i+=1

yd2 = yd + yd_model

dCoo    = dM.dot(cvm_mod).dot(dM.T)                         
dCof    = dM.dot(cvm_mod).dot(M2.T)        
A=dCof.dot(Lf_inv)
dCo = dCoo - np.dot(A,A.T)
print("Negative dCo: ",np.sum(np.diag(dCo)<0))

if np.sum(np.diag(ddCo)<0)>0 or np.sum(np.diag(dCo)<0)>0 or np.sum(np.diag(Co)<0)>0:
    print("ERROR: there are some negative variances!")

yf = dM.dot(beta_mod)
ef= np.sqrt(np.diag(dCo))

#yf2 = yf + (0.5*F2*tt*tt + tt*F1 + F0)
yf2 = yf + (0.5*F2*tt*tt)

i=0
for ge in glep:
    gt=(t-ge)*86400.0
    yf2[t>ge] += glf1[i] * gt[t>ge] + glf0[i]
    if gltd[i] > 0:
        yf2[t>ge] += glf0d[i] * np.exp(-(t[t>ge]-glep[i])/gltd[i])

    if gltd2[i] > 0:
        yf2[t>ge] += glf0d2[i] * np.exp(-(t[t>ge]-glep2[i])/gltd2[i])
    i+=1



ry += np.mean(y_dat-ry)


with open("white_ifunc.par","w") as f:
    f.writelines(inpar)
    f.write("SIFUNC 2 0\n")
    for i in range(len(t)):
        f.write("IFUNC{}  {}  {} {}\n".format(i+1,t[i],-y[i],ey[i]))



with open("nudot.asc","w") as f:
    for i in range(len(yd)):
        f.write("{} {} {} {} {}\n".format(t[i],yd[i],yd_model[i],yd2[i],ed[i]))

with open("deltanu.asc","w") as f:
    for i in range(len(yf2)):
        f.write("{} {}\n".format(t[i],yf2[i]))





fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(311)

plt.errorbar(dat_t,ry2,yerr=re2,color='k',marker='.',ls='None',ms=3.0,alpha=0.7)

plt.title("PSR "+psrn)
plt.xlabel("MJD")
plt.ylabel("residual (s)")

for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*F0)
ax2.set_ylabel("Residual (turns)")


ax = fig.add_subplot(312)
plt.plot(t,y,color='green')
plt.errorbar(dat_t,ry,yerr=dat_e,color='k',marker='.',ls='None',ms=3.0,alpha=0.7)

plt.fill_between(t,y-ey,y+ey,color='green',alpha=0.5)
#plt.title("PSR "+psrn)
plt.xlabel("MJD")
plt.ylabel("residual (s)")

for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*F0)
ax2.set_ylabel("Residual (turns)")

ax = fig.add_subplot(313)
plt.plot(t,y-y,color='green')
plt.errorbar(dat_t,ry-y_dat,yerr=dat_e,color='k',marker='.',ls='None',ms=3.0,alpha=0.7)

plt.fill_between(t,-ey,+ey,color='green',alpha=0.5)
#plt.title("PSR "+psrn)
plt.xlabel("MJD")
plt.ylabel("Residual - Model (s)")

for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*F0)
ax2.set_ylabel("Residual - Model (turns)")


plt.savefig("residuals_{}.pdf".format(psrn))

plt.figure(figsize=(16,9))
plt.plot(t,yd,color='blue')
plt.fill_between(t,yd-ed,yd+ed,color='blue',alpha=0.5)
for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

plt.title("PSR "+psrn)
plt.xlabel("MJD")
plt.ylabel("$\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)")

plt.savefig("nudot_{}.pdf".format(psrn))




fig = plt.figure(figsize=(16,9))
fig.suptitle("PSR "+psrn)

ax = fig.add_subplot(321)

plt.errorbar(dat_t,ry2,yerr=re2,color='k',marker='.',ls='None',ms=3.0,alpha=0.7)

#plt.xlabel("MJD")
plt.ylabel("residual (s)")

for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

    
ax3 = ax.twiny()
ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004)
ax3.xaxis.tick_top()
ax3.set_xlabel("Year")
ax3.xaxis.set_tick_params(direction='inout',labeltop=True)
ax.xaxis.set_tick_params(labelbottom=False,direction='in')



ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*F0)
ax2.set_ylabel("Residual (turns)")



ax = fig.add_subplot(323)
plt.plot(t,y,color='green')
plt.errorbar(dat_t,ry,yerr=dat_e,color='k',marker='.',ls='None',ms=3.0,alpha=0.7)

plt.fill_between(t,y-ey,y+ey,color='green',alpha=0.5)
#plt.title("PSR "+psrn)
#plt.xlabel("MJD")
plt.ylabel("residual (s)")

for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

    
#ax3 = ax.twiny()
#ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004)
#ax3.xaxis.tick_top()
#ax3.xaxis.set_tick_params(direction='in',labeltop=False)
ax.xaxis.set_tick_params(labelbottom=False,direction='in')
    
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*F0)
ax2.set_ylabel("Residual (turns)")


ax = fig.add_subplot(325)
plt.plot(t,y-y,color='green')
plt.errorbar(dat_t,ry-y_dat,yerr=dat_e,color='k',marker='.',ls='None',ms=3.0,alpha=0.7)

plt.fill_between(t,-ey,+ey,color='green',alpha=0.5)
#plt.title("PSR "+psrn)
plt.xlabel("MJD")
plt.ylabel("Residual - Model (s)")

for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

#ax3 = ax.twiny()
#ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004)
#ax3.xaxis.tick_top()
#ax3.xaxis.set_tick_params(direction='in',labeltop=False)
ax.xaxis.set_tick_params(labelbottom=True,direction='inout')    
    
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*F0)
ax2.set_ylabel("Residual - Model (turns)")





ax = fig.add_subplot(322)
plt.plot(t,1e6*yf2,color='orange')
plt.fill_between(t,1e6*(yf2-ef),1e6*(yf2+ef),color='orange',alpha=0.5)
for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

plt.xlabel("MJD")
plt.ylabel("$\\Delta{\\nu}$ ($\mathrm{\mu}$Hz)")
ax.yaxis.set_label_position("right")
ax.yaxis.set_tick_params(labelleft=False,labelright=True,right=True,left=False, direction='in')
ax3 = ax.twiny()
ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004)
ax3.xaxis.tick_top()
ax3.set_xlabel("Year")


ax = fig.add_subplot(324)
plt.plot(t,yd_model,color='lightblue',ls='--')
plt.plot(t,yd2,color='blue')
plt.fill_between(t,yd2-ed,yd2+ed,color='blue',alpha=0.5)

for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

plt.xlabel("MJD")
plt.ylabel("$\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)")
ax.yaxis.set_label_position("right")
ax.yaxis.set_tick_params(labelleft=False,labelright=True,right=True,left=False, direction='in')


ax = fig.add_subplot(326)
plt.plot(t,yd,color='blue')
plt.fill_between(t,yd-ed,yd+ed,color='blue',alpha=0.5)
for ge in glep:
    plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

plt.xlabel("MJD")
plt.ylabel("$\\Delta\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)")
ax.yaxis.set_label_position("right")
ax.yaxis.set_tick_params(labelleft=False,labelright=True,right=True,left=False, direction='in')


plt.subplots_adjust(hspace=0,wspace=0.15)
plt.figtext(x=0.47,y=0.94,s="$P$:{:.0f} ms".format(1000.0/F0),horizontalalignment='center')
plt.figtext(x=0.53,y=0.94,s="$\\dot{{P}}$:{:.0g}".format(-F1/F0/F0),horizontalalignment='center')
if PB > 0:
    plt.figtext(x=0.59,y=0.94,s="$P_B$:{:.1g}".format(PB),horizontalalignment='center')

plt.savefig("combined_{}.pdf".format(psrn))


plt.figure()
plt.figtext(x=0.47,y=0.94,s="$P$:{:.0f} ms".format(1000.0/F0),horizontalalignment='center')
plt.figtext(x=0.53,y=0.94,s="$\\dot{{P}}$:{:.0g}".format(-F1/F0/F0),horizontalalignment='center')
if PB > 0:
    plt.figtext(x=0.59,y=0.94,s="$P_B$:{:.1g}".format(PB),horizontalalignment='center')

plt.loglog(freqs,pwrs)
plt.title("PSR "+psrn)
plt.xlabel("Freq (yr^-1)")
plt.xlabel("Power (???)")
plt.savefig("pwrspec_{}.pdf".format(psrn))

