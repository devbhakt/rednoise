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





def doit(par,tim,par2, plots=True, write=True):

    subprocess.call(["tempo2","-output","exportres","-f",par2,tim,"-nofit","-writeres"])

    rx2,ry2,re2 = np.loadtxt("out.res",usecols=(0,5,6),unpack=True)

    if par2 == par:
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

    t=np.linspace(start-0.5,finish+0.5,1000)
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


    s2d2 = 1.0/(86400.0*86400.0)
    s2d = 1.0/86400.0

    M = np.zeros((2*nwav,len(y)))
    M2 = np.zeros((2*nwav,len(dat_t)))

    dM = np.zeros_like(M)
    ddM = np.zeros_like(M)


    if write:
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




    yd = ddM.dot(beta_mod)

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

    yf = dM.dot(beta_mod)

    yf2 = yf + (0.5*F2*tt*tt)

    i=0
    for ge in glep:
        gt=(t-ge)*86400.0
        yf2[t>ge] += glf1[i] * gt[t>ge] + glf0[i]
        if gltd[i] > 0:
            yf2[t>ge] += glf0d[i] * np.exp(-(t[t>ge]-glep[i])/gltd[i])

        if gltd2[i] > 0:
            yf2[t>ge] += glf0d2[i] * np.exp(-(t[t>ge]-glep[i])/gltd2[i])
        i+=1



    ry += np.mean(y_dat-ry)


    if write:
        with open("white_ifunc.par","w") as f:
            f.writelines(inpar)
            f.write("SIFUNC 2 0\n")
            for i in range(len(t)):
                f.write("IFUNC{}  {}  {} {}\n".format(i+1,t[i],-y[i],0))



        with open("nudot.asc","w") as f:
            for i in range(len(yd)):
                f.write("{} {} {} {}\n".format(t[i],yd[i],yd_model[i],yd2[i]))

        with open("deltanu.asc","w") as f:
            for i in range(len(yf2)):
                f.write("{} {}\n".format(t[i],yf2[i]))



    if plots:


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

        for ge in glep:
            plt.axvline(ge,linestyle="--",color='purple',alpha=0.7)

        plt.xlabel("MJD")
        plt.ylabel("$\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)")
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_tick_params(labelleft=False,labelright=True,right=True,left=False, direction='in')


        ax = fig.add_subplot(326)
        plt.plot(t,yd,color='blue')
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


    return t,yd,yd_model,yd2


par = sys.argv[1]
tim = sys.argv[2]
par2 = sys.argv[3]

if len(sys.argv) > 4:

    yds=[]
    yd_models=[]
    yd2s=[]

    samples_file = sys.argv[4]

    if len(sys.argv) > 5:
        maxsamp=int(sys.argv[5])
    else:
        maxsamp=1e99

    with open(samples_file) as f:
        pars = f.readline().split()

    for i,p in enumerate(pars):
        if p.startswith("TNEF") or p.startswith("TNEQ"):
            pars[i] = p.replace("_"," ")
    samples = np.loadtxt(samples_file,skiprows=1)
#    pars=s['pars']
#    samples=s['samples']

    with open(par) as inparfile:
        lines = inparfile.readlines()
    for isamp,s in enumerate(samples):
        if isamp > maxsamp:
            break
        with open(par+".tmp","w") as outpar:
            for line in lines:
                replaced=False
                for i,p in enumerate(pars):
                    if line.startswith(p+" "):
                        line="{} {} #r {}".format(p,s[i],line)
                outpar.write(line)
        subprocess.call(["tempo2", "-f", par+".tmp", tim, "-outpar", par+".ftmp"])
        t,yd,yd_model,yd2 = doit(par+".ftmp",tim,par+".ftmp",plots=False,write=False)
        yds.append(yd)
        yd_models.append(yd_model)
        yd2s.append(yd2)
    np.savez_compressed("nudots.npz",t=t,yds=yds,yd_models=yd_models,yd2s=yd2s)

else:
    doit(par,tim,par2)

