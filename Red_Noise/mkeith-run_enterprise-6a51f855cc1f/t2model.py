import numpy as np

from model_components import chol_red_model


def read_t2_model(file, psr, makecvm=True):
    ntoa = len(psr.toas)
    funcs = list()
    cvm = np.zeros((ntoa, ntoa))
    with open(file) as f:
        model1=False
        for line in f:
            e = line.strip().split()
            if model1:
                if e[0]=="ALPHA":
                    alpha = float(e[1])
                if e[0]=="FC":
                    fc = float(e[1])
                if e[0]=="AMP":
                    A = float(e[1])
                    if makecvm:
                        c = chol_red_model.psd2cov(psr.toas, chol_red_model.pl_red, np.log10(A), alpha, fc, fc)
                    f = lambda ff: chol_red_model.pl_red(ff, np.log10(A), alpha, fc, fc)
                    funcs.append(f)
                    model1=False
                continue
            if e[0] == "MODEL":
                if e[1] == "1":
                    model1=True
                    continue
                if e[1] == "T2":
                    continue
                c, f = parse_model_line(line, psr, makecvm)
                if makecvm:
                    cvm += c
                funcs.append(f)
            else:
                raise Exception("Sorry don't understand this model file", line,e)
    return cvm, funcs


def parse_model_line(line, psr, makecvm):
    df = 86400.0/(np.amax(psr.toas)-np.amin(psr.toas))
    e = line.split()
    if e[0] == "MODEL":
        if e[1] == "T2PowerLaw":
            alpha = float(e[2])
            A = float(e[3])
            fc = float(e[4])
            cvm = None if not makecvm else chol_red_model.psd2cov(psr.toas, chol_red_model.pl_red, np.log10(A), alpha,
                                                                  fc, fc)
            return cvm, lambda \
                    f: chol_red_model.pl_red(f, np.log10(A), alpha, fc, fc)
        elif e[1] == "T2PowerLaw_QPc":
            alpha = float(e[2])
            A = float(e[3])
            fc = float(e[4])
            log_qp_ratio = float(e[5])
            qp_f0 = float(e[6])
            sig = float(e[7])
            lam = float(e[8])
            print("sig",sig)
            print("lam",lam)
            if makecvm:
                print("MODEL T2PowerLaw_QPc",alpha,A,fc,log_qp_ratio,qp_f0,sig,lam)
            cvm = None if not makecvm else chol_red_model.psd2cov(psr.toas, chol_red_model.pl_plus_qp_nudot_cutoff,
                                                                  np.log10(A), alpha,
                                                                  log_qp_ratio, np.log10(qp_f0), sig, lam, fc, fc,df)
            return cvm, lambda \
                    f: chol_red_model.pl_plus_qp_nudot_cutoff(f, np.log10(A), alpha, log_qp_ratio, np.log10(qp_f0),
                                                              sig, lam, fc, fc,df)
        else:
            raise Exception("Sorry don't understand this model file", line)
