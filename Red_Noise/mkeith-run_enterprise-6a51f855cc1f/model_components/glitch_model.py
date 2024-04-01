import numpy as np
import sys
import bisect
from scipy.optimize import curve_fit

from enterprise.signals import deterministic_signals
from enterprise.signals import signal_base

from astropy import coordinates as coord
from astropy import units as units

from . import xparameter as parameter
from . import priors

name = "GlitchModel"
argdec = "Glitch and recovery parameters."


def setup_argparse(parser):
    parser.add_argument('--glitch-all', '--gl-all', action='store_true', help='fit for all glitches')
    parser.add_argument('--glitches', type=int, default=[], nargs='+', help='Select glitches to fit')
    parser.add_argument('--glitch-recovery', type=int, default=[], nargs='+', help='fit for glitch recoveries on these glitches')
    parser.add_argument('--glitch-double-recovery', type=int, default=[], nargs='+', help='fit for a second glitch recovery on these glitches')
    parser.add_argument('--glitch-triple-recovery', type=int, default=[], nargs='+', help='fit for a third glitch recovery on these glitches')
    parser.add_argument('--glitch-epoch-range', '--glep-range', type=float, default=0, help='Window for glitch epoch fitting')
    parser.add_argument('--glitch-td-min', '--gltd-min', type=float, default=0, help='Min log10(td)')
    parser.add_argument('--glitch-td-max', '--gltd-max', type=float, default=3, help='Max log10(td)')
    parser.add_argument('--glitch-f0-range', '--glf0-range', type=float, default=0.3, help="Fractional change in GLF0")
    parser.add_argument('--glitch-f1-range', '--glf1-range', type=float, default=0.3, help="Fractional change in GLF1")
    parser.add_argument('--glitch-f2-range', '--glf2-range', type=float, default=0, help="Fractional change in GLF2")
    parser.add_argument('--glitch-f0d-range', '--glf0d-range', type=float, default=1.0, help="Fractional range of GLF0D compared to GLF0")
    parser.add_argument('--glitch-td-range', '--gltd-range', type=float, default=None, help="Fractional change in GLTD")
    parser.add_argument('--glitch-f0d-positive', '--glf0d-positive', action='store_true', help="Allow only positive GLF0D")
    parser.add_argument('--glitch-td-split', type=float, default=[1.8, 2.8], nargs='+', help="Where to split the td prior for multi exponentials")
    parser.add_argument('--glitch-alt-f0', action='store_true', help="Use alternative parameterisation of glitches fitting for instantanious change in F0 rather than GLF0")
    parser.add_argument('--glitch-alt-f0t', type=int, default=[200]*100, nargs='+', help="Replace GLF1 with change of spin frequency 'T' days after the glitches respectively")
    parser.add_argument('--alt-f0t-gltd', action='store_true', help="Replace GLF1 with change of spin frequency 'GLTD' days after the glitch for all glitches")
    parser.add_argument('--measured-prior', action='store_true', help="Use measured prior range for GLF0(instant) and GLF0(T=taug)")
    parser.add_argument('--measured-without',action='store_true',help="Measure prior range without glitch for GLF0(instant) and GLF0(T=taug)")
    parser.add_argument('--measured-sigma', '--sigma-range', type=float, default=[30]*4, nargs='+', help="Minus/Plus sigma range of GLF0(instant), and Minus/Plus sigma range of GLF0(T=taug) respectively")
    parser.add_argument('--auto-add', action='store_true', help="Automatic add all existing glitches recoveries in the par file")

def setup_model(args, psr, parfile):
    global glitches
    glitches = {}
    for line in parfile:
        e = line.strip().split()
        if len(e) > 1:
            if e[0] == "F0":
                f0 = float(e[1])
            elif e[0] == "F1":
                f1 = float(e[1])
            elif e[0] == "F2":
                f2 = float(e[1])
            elif e[0] == "PEPOCH":
                pepoch = float(e[1])
            elif e[0].startswith("GLEP"):
                glitches[int(e[0][5:])] = {'EP': float(e[1])}
            elif e[0].startswith("GLF0D_"):
                glitches[int(e[0][6:])]['F0D'] = float(e[1])
                if args.auto_add:
                    args.glitch_recovery.append(int(e[0][6:]))
                    print("Add glitch[{}] to recovery".format(int(e[0][6:])))
            elif e[0].startswith("GLF0D2_"):
                glitches[int(e[0][7:])]['F0D2'] = float(e[1])
                if args.auto_add:
                    args.glitch_double_recovery.append(int(e[0][7:]))
                    print("Add glitch[{}] to double recovery".format(int(e[0][7:])))
            elif e[0].startswith("GLF0D3_"):
                glitches[int(e[0][7:])]['F0D3'] = float(e[1])
                if args.auto_add:
                    args.glitch_triple_recovery.append(int(e[0][7:]))
                    print("Add glitch[{}] to triple recovery".format(int(e[0][7:])))
            elif e[0].startswith("GLTD_"):
                glitches[int(e[0][5:])]['TD'] = float(e[1])
            elif e[0].startswith("GLTD2_"):
                glitches[int(e[0][6:])]['TD2'] = float(e[1])
            elif e[0].startswith("GLTD3_"):
                glitches[int(e[0][6:])]['TD3'] = float(e[1])
            elif e[0].startswith("GLF0_"):
                glitches[int(e[0][5:])]['F0'] = float(e[1])
            elif e[0].startswith("GLF1_"):
                glitches[int(e[0][5:])]['F1'] = float(e[1])
            elif e[0].startswith("GLF2_"):
                glitches[int(e[0][5:])]['F2'] = float(e[1])

        for gl in args.glitches:
            if "GLF0_{}".format(gl) in psr.fitpars or "GLF1_{}".format(gl) in psr.fitpars:
                print("ERROR: Can't fit for GLF0/GLF1 in both MCMC and least-squares")
                sys.exit(1)
            if gl in args.glitch_recovery:
                if "GLF0D_{}".format(gl) in psr.fitpars or "GLTD_{}".format(gl) in psr.fitpars:
                    print("ERROR: Can't fit for GLF0D/GLTD in both MCMC and least-squares")
                    sys.exit(1)
                if "GLF0D2_{}".format(gl) in psr.fitpars or "GLTD2_{}".format(gl) in psr.fitpars:
                    print("ERROR: Can't fit for GLF0D2/GLTD2 in both MCMC and least-squares")
                    sys.exit(1)
                if "GLF0D3_{}".format(gl) in psr.fitpars or "GLTD3_{}".format(gl) in psr.fitpars:
                    print("ERROR: Can't fit for GLF0D3/GLTD3 in both MCMC and least-squares")
                    sys.exit(1)

    components = []
    if args.glitches is None:
        args.glitches=[]
    if args.glitch_recovery is None:
        args.glitch_recovery = []

    glitches = dict(sorted(glitches.items()))

    for gi, gl in enumerate(glitches):
        if args.glitch_all or gl in args.glitches: # or gl in args.glitch_recovery:
            ## pre-compute the model for the existing glitch
            glf0=glf1=glf2=glf0d=glf0d2=glf0d3=0
            gltd=gltd2=gltd3=200
            glf2fit = False
            glep = glitches[gl]['EP']
            if 'F0' in glitches[gl]:
                glf0 = glitches[gl]['F0']
            if 'F1' in glitches[gl]:
                glf1 = glitches[gl]['F1']
            if 'F2' in glitches[gl]:
                glf2 = glitches[gl]['F2']
                glf2fit = True
            if 'F0D' in glitches[gl]:
                glf0d = glitches[gl]['F0D']
            if 'TD' in glitches[gl]:
                gltd = glitches[gl]['TD']
            if 'F0D2' in glitches[gl]:
                glf0d2 = glitches[gl]['F0D2']
            if 'TD2' in glitches[gl]:
                gltd2 = glitches[gl]['TD2']
            if 'F0D3' in glitches[gl]:
                glf0d2 = glitches[gl]['F0D3']
            if 'TD3' in glitches[gl]:
                gltd3 = glitches[gl]['TD3']
            # This is the 'old model'
            glitches[gl]['old_model'] = model_glitch(psr.toas, f0, glep, glf0, glf1, glf2, glf0d, gltd, glf0d2, gltd2, glf0d3, gltd3)
    
            if args.glitch_alt_f0t:
                if args.alt_f0t_gltd:
                    glf0t = int(min(gltd,gltd2,gltd3)) # set tau_g to the minimum exponential time scale
                elif gi<len(args.glitch_alt_f0t):
                    glf0t = int(args.glitch_alt_f0t[gi]) # set tau_g according to input values(measured from ToAs by hand)
                else:
                    glf0t = 200 # set tau_g to some default value
                if args.measured_prior:
                    k_pre, k_post, k_p2, k_200 = slope_range(psr.toas, f0*psr.residuals, psr.toaerrs, glep, glf0t, np.abs(args.measured_sigma))
                    if args.measured_without:
                        glf0_i_range, glf0_t_range = prior_range(k_pre, k_post, k_p2, k_200, glf0t, None, None, glf0d, gltd, glf0d2, gltd2, glf0d3, gltd3, glf2)
                    else:
                        glf0_i_range, glf0_t_range = prior_range(k_pre, k_post, k_p2, k_200, glf0t, glf0, glf1, glf0d, gltd, glf0d2, gltd2, glf0d3, gltd3, glf2)
                    print("Measured ranges for GLF0(instant):", glf0_i_range)
                    print("Measured ranges for GLF0(T=%d):"%glf0t, glf0_t_range)
                    del glf0_i_range[1], glf0_t_range[1]

            if args.glitch_epoch_range > 0:
                glep = parameter.Uniform(glitches[gl]['EP']-args.glitch_epoch_range, glitches[gl]['EP']+args.glitch_epoch_range, to_par=to_par)("GLEP_{}".format(gl))
                priors.glep_range[gi] = [glitches[gl]['EP']-args.glitch_epoch_range, glitches[gl]['EP']+args.glitch_epoch_range]
            else:
                glep=parameter.Constant(glitches[gl]['EP'])
                priors.glep_range[gi] = glitches[gl]['EP']

            glf0_range = sorted([glf0 - args.glitch_f0_range*glf0, glf0 + args.glitch_f0_range*glf0])
            glf1_range = sorted([glf1 - args.glitch_f1_range*glf1, glf1 + args.glitch_f1_range*glf1])
            glf0d_range = [-args.glitch_f0d_range*abs(glf0)+glf0d, args.glitch_f0d_range*abs(glf0)+glf0d]
            glf0d2_range = [-args.glitch_f0d_range*abs(glf0)+glf0d2, args.glitch_f0d_range*abs(glf0)+glf0d2]
            glf0d3_range = [-args.glitch_f0d_range*abs(glf0)+glf0d3, args.glitch_f0d_range*abs(glf0)+glf0d3]

            if args.glitch_f0d_positive and glf0d_range[0]<0:
                glf0d_range[0]=0

            if glf2fit is False or args.glitch_f2_range is None or args.glitch_f2_range==0:
                glf2_range = 0
                glf2 = parameter.Constant(0)
            elif glf2==0:
                glf2_range = sorted([glf2 - args.glitch_f2_range * f2, glf2 + args.glitch_f2_range * f2])
                glf2 = parameter.Uniform(glf2_range[0]*1e18, glf2_range[1]*1e18,to_par=to_par)("GLF2_{}".format(gl))
            else:
                glf2_range = sorted([glf2 - args.glitch_f2_range * glf2, glf2 + args.glitch_f2_range * glf2])
                glf2 = parameter.Uniform(glf2_range[0]*1e18, glf2_range[1]*1e18,to_par=to_par)("GLF2_{}".format(gl))
            priors.glf2_range[gi] = glf2_range

            if args.glitch_alt_f0: # Alternate parameterisation for GLF0
                if args.measured_prior:
                    glf0_alt_range = glf0_i_range
                    glf0_alt = parameter.Uniform(glf0_alt_range[0] * 1e6, glf0_alt_range[1] * 1e6, to_par=to_par)("GLF0(instant)_{}".format(gl))
                else:
                    glf0_alt_range = [glf0_range[0] + glf0d_range[0] + glf0d2_range[0] + glf0d3_range[0], glf0_range[1] + glf0d_range[1] + glf0d2_range[1] + glf0d3_range[1]]
                    glf0_alt = parameter.Uniform(glf0_alt_range[0] * 1e6, glf0_alt_range[1] * 1e6, to_par=to_par)("GLF0(instant)_{}".format(gl))
                priors.glf0i_range[gi] = glf0_alt_range
                glf0_range = 0
                glf0 = None
            else:
                glf0_alt = None
                glf0 = parameter.Uniform(glf0_range[0] * 1e6, glf0_range[1] * 1e6, to_par=to_par)(
                    "GLF0_{}".format(gl))  # units of 10^-6


            if args.glitch_alt_f0 and args.glitch_alt_f0t: # Alternate parameterisation for GLF1
                if args.measured_prior:
                    glf0_at_t_range = glf0_t_range
                    glf0_at_t = parameter.Uniform(glf0_at_t_range[0] * 1e6, glf0_at_t_range[1] * 1e6, to_par=to_par)("GLF0(T={:d})_{}".format(glf0t,gl))  # units of 10^-6
                else:
                    glf0_at_t_range = [glf0t * 86400 * glf1_range[0] - glf0d_range[1] - glf0d2_range[1] - glf0d3_range[1], glf0t * 86400 * glf1_range[1]] # Fix GLF2, priors doesn't matter
                    glf0_at_t = parameter.Uniform(glf0_at_t_range[0] * 1e6, glf0_at_t_range[1] * 1e6, to_par=to_par)("GLF0(T={:d})_{}".format(glf0t,gl))  # units of 10^-6
                priors.glf0t_range[gi] = glf0_at_t_range
                glf1_range = 0
                glf1 = None
                taug = glf0t
                glf0t = parameter.Constant(glf0t)
            else:
                glf0_at_t = None
                taug = 0
                glf0t = None
                glf1 = parameter.Uniform(glf1_range[0]*1e12,glf1_range[1]*1e12,to_par=to_par)("GLF1_{}".format(gl))  # units of 10^-12

            if gl in args.glitch_recovery or gl in args.glitch_double_recovery or gl in args.glitch_triple_recovery:
                f0d = parameter.Uniform(glf0d_range[0]*1e6,glf0d_range[1]*1e6,to_par=to_par)("GLF0D_{}".format(gl))  # units of 10^-6
                if gl in args.glitch_triple_recovery:
                    if args.glitch_td_range is None:
                        gltd_range = sorted([args.glitch_td_min, args.glitch_td_split[0]])
                        gltd2_range = sorted([args.glitch_td_split[0], args.glitch_td_split[1]])
                        gltd3_range = sorted([args.glitch_td_split[1], args.glitch_td_max])
                        td = parameter.Uniform(gltd_range[0], gltd_range[1], to_par=to_par)("GLTD_{}".format(gl))
                        f0d_2 = parameter.Uniform(glf0d2_range[0]*1e6, glf0d2_range[1]*1e6, to_par=to_par)(
                        "GLF0D2_{}".format(gl))  # units of 10^-6
                        td_2 = parameter.Uniform(gltd2_range[0], gltd2_range[1], to_par=to_par)("GLTD2_{}".format(gl))
                        f0d_3 = parameter.Uniform(glf0d3_range[0]*1e6, glf0d3_range[1]*1e6, to_par=to_par)(
                        "GLF0D3_{}".format(gl))  # units of 10^-6
                        td_3 = parameter.Uniform(gltd3_range[0], gltd3_range[1], to_par=to_par)("GLTD3_{}".format(gl))
                    else:
                        gltd_range = sorted([gltd+np.log10(1-args.glitch_td_range), gltd+np.log10(1+args.glitch_td_range)])
                        gltd2_range = sorted([gltd2+np.log10(1-args.glitch_td_range), gltd2+np.log10(1+args.glitch_td_range)])
                        gltd3_range = sorted([gltd3+np.log10(1-args.glitch_td_range), gltd3+np.log10(1+args.glitch_td_range)])
                        td = parameter.Uniform(gltd_range[0], gltd_range[1], to_par=to_par)("GLTD_{}".format(gl))
                        f0d_2 = parameter.Uniform(glf0d2_range[0]*1e6, glf0d2_range[1]*1e6, to_par=to_par)(
                        "GLF0D2_{}".format(gl))  # units of 10^-6
                        td_2 = parameter.Uniform(gltd2_range[0], gltd2_range[1], to_par=to_par)("GLTD2_{}".format(gl))
                        f0d_3 = parameter.Uniform(glf0d3_range[0]*1e6, glf0d3_range[1]*1e6, to_par=to_par)(
                        "GLF0D3_{}".format(gl))  # units of 10^-6
                        td_3 = parameter.Uniform(gltd3_range[0], gltd3_range[1], to_par=to_par)("GLTD3_{}".format(gl))
                elif gl in args.glitch_double_recovery:
                    if args.glitch_td_range is None:
                        gltd_range = sorted([args.glitch_td_min, args.glitch_td_split[0]])
                        gltd2_range = sorted([args.glitch_td_split[0], args.glitch_td_max])
                        td = parameter.Uniform(gltd_range[0], gltd_range[1], to_par=to_par)("GLTD_{}".format(gl))
                        f0d_2 = parameter.Uniform(glf0d2_range[0]*1e6, glf0d2_range[1]*1e6, to_par=to_par)(
                        "GLF0D2_{}".format(gl))  # units of 10^-6
                        td_2 = parameter.Uniform(gltd2_range[0], gltd2_range[1], to_par=to_par)("GLTD2_{}".format(gl))
                    else:
                        gltd_range = sorted([gltd+np.log10(1-args.glitch_td_range), gltd+np.log10(1+args.glitch_td_range)])
                        gltd2_range = sorted([gltd2+np.log10(1-args.glitch_td_range), gltd2+np.log10(1+args.glitch_td_range)])
                        td = parameter.Uniform(gltd_range[0], gltd_range[1], to_par=to_par)("GLTD_{}".format(gl))
                        f0d_2 = parameter.Uniform(glf0d2_range[0]*1e6, glf0d2_range[1]*1e6, to_par=to_par)(
                        "GLF0D2_{}".format(gl))  # units of 10^-6
                        td_2 = parameter.Uniform(gltd2_range[0], gltd2_range[1], to_par=to_par)("GLTD2_{}".format(gl))
                    glf0d3_range = 0
                    gltd3_range = 0
                    f0d_3 = parameter.Constant(0)
                    td_3 = parameter.Constant(1)
                else:
                    if args.glitch_td_range is None:
                        gltd_range = sorted([args.glitch_td_min, args.glitch_td_max])
                        td = parameter.Uniform(args.glitch_td_min, args.glitch_td_max,to_par=to_par)("GLTD_{}".format(gl))
                    else:
                        gltd_range = sorted([gltd+np.log10(1-args.glitch_td_range), gltd+np.log10(1+args.glitch_td_range)])
                        td = parameter.Uniform(gltd_range[0], gltd_range[1],to_par=to_par)("GLTD_{}".format(gl))
                    glf0d2_range = 0
                    gltd2_range = 0
                    glf0d3_range = 0
                    gltd3_range = 0
                    f0d_2 = parameter.Constant(0)
                    td_2 = parameter.Constant(1)
                    f0d_3 = parameter.Constant(0)
                    td_3 = parameter.Constant(1)
            else:
                glf0d_range = 0
                gltd_range = 0
                glf0d2_range = 0
                gltd2_range = 0
                glf0d3_range = 0
                gltd3_range = 0
                f0d = parameter.Constant(0)
                td = parameter.Constant(1)
                f0d_2 = parameter.Constant(0)
                td_2 = parameter.Constant(1)
                f0d_3 = parameter.Constant(0)
                td_3 = parameter.Constant(1)
            priors.glf0_range[gi] = glf0_range
            priors.glf1_range[gi] = glf1_range
            priors.glf0d_range[gi] = glf0d_range
            priors.gltd_range[gi] = np.power(10, gltd_range)
            priors.glf0d2_range[gi] = glf0d2_range
            priors.gltd2_range[gi] = np.power(10, gltd2_range)
            priors.glf0d3_range[gi] = glf0d3_range
            priors.gltd3_range[gi] = np.power(10, gltd3_range)
            priors.taug[gi] = taug
            print_prior_ranges(gl)

            glexp = model_glitch_residual(f0=parameter.Constant(f0), glep=glep,
                                          glf0=glf0, glf1=glf1, glf0d=f0d, log10_td=td,
                                          glitch_id=parameter.Constant(gl), glf2=glf2,
                                          glf0d2 = f0d_2, log10_td2=td_2, glf0d3 = f0d_3, log10_td3=td_3, 
                                          glf0_alt=glf0_alt, glf0_at_t=glf0_at_t, glf0t=glf0t)

            glexp_f = deterministic_signals.Deterministic(glexp, name="gl{}:".format(gl))
            components.append(glexp_f)

    if len(components) > 0:
        model = components[0]
        for m in components[1:]:
            model += m
        return model
    else:
        return None


def print_prior_ranges(gl=None):
    global glitches
    if isinstance(gl, int) and 0 < gl <= len(glitches):
        gi = gl -1
        print("Prior information for glitch {}:".format(gl))
        print("GLEP range", priors.glep_range[gi])
        print("GLF0 range", priors.glf0_range[gi])
        print("GLF1 range", priors.glf1_range[gi])
        print("GLF2 range", priors.glf2_range[gi])
        print("GLF0(instant) range", priors.glf0i_range[gi])
        print("GLF0(T={}) range".format(priors.taug[gi]), priors.glf0t_range[gi])
        print("GLF0D range", priors.glf0d_range[gi])
        print("GLTD range", priors.gltd_range[gi])
        print("GLF0D2 range", priors.glf0d2_range[gi])
        print("GLTD2 range", priors.gltd2_range[gi])
        print("GLF0D3 range", priors.glf0d3_range[gi])
        print("GLTD3 range", priors.gltd3_range[gi])
        print("")
    else:
        for gi, gl in enumerate(glitches):
            print_prior_ranges(gl)


def to_par(self, p, chain):
    if p.startswith("GL"):
        pname=p
        if "GLF0" in pname:
            return pname, chain * 1e-6
        elif "GLF1" in pname:
            return pname, chain * 1e-12
        elif "GLF2" in pname:
            return pname, chain * 1e-18
        elif "GLTD" in pname:
            return pname, 10**chain
        else:
            return pname,chain
    else:
        return p,chain


def model_glitch(toas, f0, glep, glf0=0, glf1=0, glf2=0, glf0d=0, gltd=1, glf0d2=0, gltd2=1, glf0d3=0, gltd3=1):
    t = toas - glep * 86400.0
    m = t >= 0
    glif = np.zeros_like(toas)
    glif[m] = glf0 * t[m] + 0.5 * glf1 * (t[m]) ** 2 + glf2 * t[m] ** 3 / 6.0
    phs = -glif
    expf = np.zeros_like(toas) + 1.0  # Cancel recovery when t<0
    expf[m] = np.exp(-t[m] / (gltd * 86400.0))  # No +1.0!
    expf2 = np.zeros_like(toas) + 1.0  # Cancel recovery when t<0
    expf2[m] = np.exp(-t[m] / (gltd2 * 86400.0))  # No +1.0!
    expf3 = np.zeros_like(toas) + 1.0  # Cancel recovery when t<0
    expf3[m] = np.exp(-t[m] / (gltd3 * 86400.0))  # No +1.0!
    phs += - glf0d * gltd * 86400 * (1.0 - expf)
    phs += - glf0d2 * gltd2 * 86400 * (1.0 - expf2)
    phs += - glf0d3 * gltd3 * 86400 * (1.0 - expf3)
    return phs / f0


@signal_base.function
def model_glitch_residual(toas, f0, glep, glf0=0, glf1=0, glf2=0, glf0d=0, log10_td=0, glitch_id=1, glf0d2=0, log10_td2=0, glf0d3=0, log10_td3=0, glf0_alt=None, glf0_at_t=None, glf0t=None):
    global glitches
    ## add conversion for alternative glitch parameters
    if not glf0_alt is None:
        glf0 = (glf0_alt - glf0d - glf0d2 - glf0d3)
    if not glf0t is None:
        ## note glf0_at_t and glf0d are in units of 1e-6... glf1 should be in units of 1e-12
        glf1 = 1e6*(glf0_at_t - glf0d*(np.exp(-glf0t/10**log10_td)-1) - glf0d2*(np.exp(-glf0t/10**log10_td2)-1) - glf0d3*(np.exp(-glf0t/10**log10_td3)-1)) / (glf0t*86400.0) - 0.5e-6*glf2*(glf0t*86400)
    ret = model_glitch(toas, f0=f0, glep=glep, glf0=glf0*1e-6, glf1=glf1*1e-12, glf2=glf2*1e-18, glf0d=glf0d*1e-6, gltd=10**log10_td, glf0d2=glf0d2*1e-6, gltd2=10**log10_td2, glf0d3=glf0d3*1e-6, gltd3=10**log10_td3) # add 2nd and 3rd recovery if needed
    ret -= glitches[int(glitch_id)]['old_model'] # remove old glitch
    return ret

# mod ly
def slope_range(toas, phs, err, glep, glf0t, sigma):
    def lin(x, a, b):
        return a*x+b

    def qua(x, a, b, c):
        return a*x**2+b*x+c

    while len(sigma)<4:
        sigma = np.append(sigma, None)
    if sigma[1] is None:
        sigma[1]=sigma[0]
    if sigma[2] is None:
        sigma[2]=sigma[0]
    if sigma[3] is None:
        sigma[3]=sigma[2]

    idx_gli = bisect.bisect_left(toas, glep*86400)
    idx_200 = bisect.bisect_left(toas, (glep+glf0t)*86400)
    idx_bgn = max(idx_gli-5, 0) # Ensure non-negative index
    toas_pre = toas[idx_bgn:idx_gli] # in seconds
    toas_post = toas[idx_gli:idx_gli+5] # in seconds
    toas_200 = toas[idx_200-2:idx_200+3] # in seconds
    phs_pre = phs[idx_bgn:idx_gli]
    phs_post = phs[idx_gli:idx_gli+5]
    phs_200 = phs[idx_200-2:idx_200+3]
    err_pre = err[idx_bgn:idx_gli]
    err_post = err[idx_gli:idx_gli+5]
    err_200 = err[idx_200-2:idx_200+3]
    opt_pre, cov_pre = curve_fit(lin, toas_pre, phs_pre, sigma=err_pre)
    opt_post, cov_post = curve_fit(lin, toas_post, phs_post, sigma=err_post)
    opt_200, cov_200 = curve_fit(lin, toas_200, phs_200, sigma=err_200)
    dev_pre = np.sqrt(np.diag(cov_pre))
    dev_post = np.sqrt(np.diag(cov_post))
    dev_200 = np.sqrt(np.diag(cov_200))
    k_pre = [opt_pre[0]-sigma[0]*dev_pre[0], opt_pre[0], opt_pre[0]+sigma[1]*dev_pre[0]]
    k_post = [opt_post[0]-sigma[1]*dev_post[0], opt_post[0], opt_post[0]+sigma[0]*dev_post[0]]
    k_p2 = [opt_post[0]-sigma[2]*dev_post[0], opt_post[0], opt_post[0]+sigma[3]*dev_post[0]]
    k_200 = [opt_200[0]-sigma[3]*dev_200[0], opt_200[0], opt_200[0]+sigma[2]*dev_200[0]]
    return k_pre, k_post, k_p2, k_200


def prior_range(k_pre, k_post, k_p2, k_200, glf0t, glf0, glf1, glf0d, gltd, glf0d2=0, gltd2=1, glf0d3=0, gltd3=1, glf2=0):
    gi = sorted([k_pre[0]-k_post[-1], k_pre[1]-k_post[1], k_pre[-1]-k_post[0]])
    gt = sorted([k_p2[0]-k_200[-1], k_p2[1]-k_200[1], k_p2[-1]-k_200[0]])
    print("Delta_k gi:", gi)
    print("Delta_k gt:", gt)    
    gpara = [glf0, glf1, glf0d, gltd]
    if all(para is not None for para in gpara):
        glf0_i = [gli + glf0 + glf0d + glf0d2 + glf0d3 for gli in gi]
        #glf0_t = [glt + glf1*glf0t*86400 + glf0d*(np.exp(-glf0t/gltd)-1) + glf0d2*(np.exp(-glf0t/gltd2)-1) + glf0d3*(np.exp(-glf0t/gltd3)-1) for glt in gt] # Fix GLF2, doesn't matter
        glf0_t = [glt + glf1*glf0t*86400 + 0.5*glf2*(glf0t*86400)**2 + glf0d*(np.exp(-glf0t/gltd)-1) + glf0d2*(np.exp(-glf0t/gltd2)-1) + glf0d3*(np.exp(-glf0t/gltd3)-1) for glt in gt]
        return glf0_i, glf0_t #Bug? no need to add these in priors?
    else:
        return gi, gt

