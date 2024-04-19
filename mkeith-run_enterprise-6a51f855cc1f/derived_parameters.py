import numpy as np
import numpy.polynomial as polynomial


def compute_derived_parameters(args,psr,par,parfile,scaled_samples,log_prob,weights,parnames):
    derived=[]
    derived_names=[]

    for i,p in enumerate(parnames):
        if p.startswith("GLF0(instant)"):
            iglitch = int(p.split('_')[1])
            glf0ins = np.copy(scaled_samples[:,i])
            glf0 = np.copy(scaled_samples[:,i]) ## no glitch recovery so GLF0=GLF0(instant)
            if "GLF0D_{}".format(iglitch) in parnames:
                glf0d = scaled_samples[:, parnames=="GLF0D_{}".format(iglitch)][:, 0]
                glf0 -= glf0d
                q = glf0d/glf0ins
                if "GLF0D2_{}".format(iglitch) in parnames:
                    glf0d2 = scaled_samples[:, parnames=="GLF0D2_{}".format(iglitch)][:, 0]
                    glf0 -= glf0d2
                    q += glf0d2/glf0ins
                    q1 = glf0d/glf0ins
                    q2 = glf0d2/glf0ins
                    derived.append(q1)
                    derived_names.append("Q1_{}".format(iglitch))
                    derived.append(q2)
                    derived_names.append("Q2_{}".format(iglitch))
                if "GLF0D3_{}".format(iglitch) in parnames:
                    glf0d3 = scaled_samples[:, parnames=="GLF0D3_{}".format(iglitch)][:, 0]
                    glf0 -= glf0d3
                    q += glf0d3/glf0ins
                    q3 = glf0d3/glf0ins
                    derived.append(q3)
                    derived_names.append("Q3_{}".format(iglitch))
                derived.append(q)
                derived_names.append("Q_{}".format(iglitch))
            derived.append(glf0)
            derived_names.append("GLF0_{}".format(iglitch))

        if p.startswith("GLF0(T="):
            iglitch = int(p.split('_')[1])
            numb = p.split('=')[1]
            glaltf0t = int(numb.split(')')[0])
            #glaltf0t = int(args.glitch_alt_f0t)
            glf0_at_t = np.copy(scaled_samples[:,i])
            if "GLF2_{}".format(iglitch) in parnames:
                glf2 =scaled_samples[:, parnames=="GLF2_{}".format(iglitch)][:,0]
            else:
                glf2=0
            if "GLF0D_{}".format(iglitch) in parnames:
                glf0d =scaled_samples[:, parnames=="GLF0D_{}".format(iglitch)][:,0]
                gltd = scaled_samples[:, parnames=="GLTD_{}".format(iglitch)][:, 0]
            else:
                glf0d=0
                gltd=1
            if "GLF0D2_{}".format(iglitch) in parnames:
                glf0d2 =scaled_samples[:, parnames=="GLF0D2_{}".format(iglitch)][:,0]
                gltd2 = scaled_samples[:, parnames=="GLTD2_{}".format(iglitch)][:, 0]
            else:
                glf0d2=0
                gltd2=1
            if "GLF0D3_{}".format(iglitch) in parnames:
                glf0d3 =scaled_samples[:, parnames=="GLF0D3_{}".format(iglitch)][:,0]
                gltd3 = scaled_samples[:, parnames=="GLTD3_{}".format(iglitch)][:, 0]
            else:
                glf0d3=0
                gltd3=1
            glf1 = (glf0_at_t - glf0d*(np.exp(-glaltf0t/gltd)-1) - glf0d2*(np.exp(-glaltf0t/gltd2)-1) - glf0d3*(np.exp(-glaltf0t/gltd3)-1)) / (glaltf0t*86400.0) - 0.5*glf2*(glaltf0t*86400)
            derived.append(glf1)
            derived_names.append("GLF1_{}".format(iglitch))
            glf1ins = glf1 - glf0d/(86400*gltd) - glf0d2/(86400*gltd2) - glf0d3/(86400*gltd3)
            derived.append(glf1ins)
            derived_names.append("GLF1(instant)_{}".format(iglitch))

        if p.startswith("T2Chol_QpF0"):
            derived.append(10**-scaled_samples[:,i])
            derived_names.append("T2Chol_QpPeriod")
        if p.startswith("T2Chol_QpPeriod"):
            derived.append(-np.log10(scaled_samples[:, i]))
            derived_names.append("T2Chol_QpF0")
        if p.startswith("TN_QpF0"):
            derived.append(10 ** -scaled_samples[:, i])
            derived_names.append("TN_QpPeriod")
        if p.startswith("TN_QpPeriod"):
            derived.append(-np.log10(scaled_samples[:, i]))
            derived_names.append("TN_QpF0")

    if args.legendre:
        l1=l2=l3=0
        if "L1" in parnames:
            l1=scaled_samples[:,parnames=="L1"][:,0]
        else:
            l1 = np.zeros(scaled_samples.shape[0])
        if "L2" in parnames:
            l2 = scaled_samples[:, parnames=="L2"][:,0]
        else:
            l2 = np.zeros(scaled_samples.shape[0])
        if "L3" in parnames:
            l3 = scaled_samples[:, parnames=="L3"][:,0]
        else:
            l3=np.zeros_like(l1)
        domain = [psr.toas[0],psr.toas[-1]]

        for line in parfile:
            e = line.strip().split()
            if len(e) > 1:
                if e[0] == "F0":
                    f0 = float(e[1])


        print("XXX ",scaled_samples.shape,l1.shape,l2.shape,l3.shape)

        xxx = polynomial.legendre.Legendre([0,0,0,6],domain=domain,window=[-1,1])
        poly = xxx.convert(kind=polynomial.polynomial.Polynomial, domain=domain,window=domain)
        print(poly.coef*f0*np.array([1,1,2,6]))
        nsamples=len(l1)
        df0=np.zeros_like(l1)
        df1=np.zeros_like(l1)
        df2=np.zeros_like(l1)
        for s in range(nsamples):
            leg = polynomial.legendre.Legendre([0,l1[s],l2[s],l3[s]],domain=domain,window=[-1,1])
            poly = leg.convert(kind=polynomial.polynomial.Polynomial, domain=domain,window=domain)
            df0[s] = poly.coef[1]*f0
            df1[s] = 2*poly.coef[2]*f0
            if "L3" in parnames:
                df2[s] = 6*poly.coef[3]*f0

        derived.append(df0)
        derived.append(df1)
        if "L3" in parnames:
            derived.append(df2)
        derived_names.append("dF0")
        derived_names.append("dF1")
        if "L3" in parnames:
            derived_names.append("F2")

    ## Check if plot_Ared_at_T is in the args
    if 'plot_Ared_at_T' in vars(args):
        if not args.plot_Ared_at_T is None:
            logAred = scaled_samples[:, parnames=="TNRedAmp"][:, 0]
            gamma = scaled_samples[:, parnames=="TNRedGam"][:, 0]
            fref = 1/args.plot_Ared_at_T
            PredT = 2*logAred - np.log10(12*np.pi*np.pi) - gamma*np.log10(fref)
            derived.append(PredT)
            derived_names.append("RedP_T{}".format(args.plot_Ared_at_T))


    return np.array(derived).T,derived_names




