from . import xparameter as parameter
import numpy as np
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise.signals.parameter import function
import enterprise.constants as const

from enterprise.signals import deterministic_signals

import sys

name = "RedNoise"
argdec = "RedNoise Parameters. Note that the red noise model is default ENABLED"


def setup_argparse(parser):
    parser.add_argument('--no-red-noise', dest='red', default=True, action='store_false',
                        help='Disable Power Law Red Noise search')
    parser.add_argument('--Ared-max', '-A', type=float, default=-12, help='Max log10A_Red')
    parser.add_argument('--Ared-min', type=float, default=-18, help='Min log10A_Red')
    parser.add_argument('--red-gamma-max', type=float, default=10, help='Max gamma red')
    parser.add_argument('--red-gamma-min', type=float, default=0, help='Min gamma red')
    parser.add_argument('--red-prior-log', action='store_true',
                        help='Use uniform prior in log space for red noise amplitude')
    parser.add_argument('--red-ncoeff', type=int, default=60, help='Number of red noise coefficients (nC)')
    parser.add_argument('--red-minf', type=float, default=None, help='Low frequency cut-off')
    parser.add_argument('--tspan-mult', type=float, default=2, help='Multiplier for tspan')
    parser.add_argument('--qp', action='store_true', help='Use QP nudot model')
    # parser.add_argument('--qp-f0-min', type=float, default=None)
    # parser.add_argument('--qp-f0-max', type=float, default=None)
    parser.add_argument('--qp-ratio-max', type=float, default=3.5)
    parser.add_argument('--qp-sigma-max', type=float, default=0.2)

    parser.add_argument('--qp-p-min-np', type=float, default=4.0)
    parser.add_argument('--qp-p-min', type=float, default=None)
    parser.add_argument('--qp-p-max', type=float, default=None)

    parser.add_argument('--plot-Ared-at-T', type=float, default=None, help="Plot derived Ared at freq 1/(T yr)")


def setup_model(args, psr, parfile):
    if args.red:
        # We scale Tspan if we are using tspan-mult
        nC = args.red_ncoeff
        Tspan = psr.toas.max() - psr.toas.min()

        Tspan *= args.tspan_mult
        nC *= int(args.tspan_mult)

        # Add the nC and flow parameters to the par file
        parfile.append("TNRedC {}\n".format(nC))
        parfile.append("TNRedFLow {}\n".format(np.log10(1.0 / args.tspan_mult)))

        # red noise (powerlaw with nC frequencies)
        if args.red_prior_log:
            log10_A = parameter.Uniform(args.Ared_min, args.Ared_max, to_par=to_par)
        else:
            log10_A = parameter.LinearExp(args.Ared_min, args.Ared_max, to_par=to_par)
        gamma = parameter.Uniform(args.red_gamma_min, args.red_gamma_max, to_par=to_par)

        if args.qp:
            tspan_days = (np.amax(psr.toas) - np.amin(psr.toas)) / 86400.0
            avg_cadence = tspan_days / len(psr.toas)

            # qp_f0_min = np.log10(2.0 / tspan_days) if args.qp_f0_min is None else args.qp_f0_min
            # qp_f0_max = np.log10(min(0.5 / avg_cadence, 1 / 20.0)) if args.qp_f0_max is None else args.qp_f0_max

            fmin = nC / tspan_days / args.tspan_mult
            qp_p_max = tspan_days / args.qp_p_min_np if args.qp_p_max is None else args.qp_p_max
            qp_p_min = max(2 * avg_cadence, 1.2/fmin) if args.qp_p_min is None else args.qp_p_min
            qp_f0_max = np.log10(1.0 / qp_p_min)  ## for below calculation
            if 10 ** qp_f0_max > fmin:
                fstep = 1.0 / tspan_days / args.tspan_mult
                needC = (10 ** qp_f0_max) / fstep
                print("ERROR: Not enough components for QP model QPF0 max is {}, fmin is {}".format(10 ** qp_f0_max,
                                                                                                    fmin))
                print("       Would need nC={}".format(needC))
                sys.exit(1)

            log_qp_ratio = parameter.Uniform(-2, args.qp_ratio_max, to_par=to_par)
            # log_f0 = parameter.Uniform(qp_f0_min, qp_f0_max, to_par=to_par)
            qp_period = parameter.Uniform(qp_p_min, qp_p_max, to_par=to_par)
            sig = parameter.Uniform(1e-3, args.qp_sigma_max, to_par=to_par)
            lam = parameter.Uniform(0.01, 10, to_par=to_par)
            df = 86400.0 / (np.amax(psr.toas) - np.amin(psr.toas))
            df = parameter.Constant(df)
            pl = powerlaw_qp(log10_A=log10_A, gamma=gamma, log_qp_ratio=log_qp_ratio, p=qp_period, sig=sig, lam=lam,
                             fmin=df)
        else:
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)

        modes=None
        fmodes=None
        if not (args.red_minf is None):
            modes = 1.0 * np.arange(1, nC + 1) / Tspan
            sectoyr=86400.0*365.25
            fmodes=modes[modes<=args.red_minf/sectoyr]
            modes = modes[modes>args.red_minf/sectoyr]
            print("Now have {} fourier components above {} yr^-1".format(len(modes),args.red_minf))

            if len(modes) < 1:
                print("Not enough modes! ",(1.0 * np.arange(1, nC + 1) / Tspan)[0])
                sys.exit(1)
            nC = len(modes)

        rn = gp_signals.FourierBasisGP(spectrum=pl, components=nC, Tspan=Tspan, modes=modes)

        if not fmodes is None:
            for i,f in enumerate(fmodes):
                A = parameter.Uniform(-50, 50,to_par=to_par)("fsinA{}".format(i))
                B = parameter.Uniform(-50, 50,to_par=to_par)("fcosB{}".format(i))
                rn += deterministic_signals.Deterministic(sinusoid(A=A,B=B,om=2*np.pi*f))

        return rn
    else:
        return None


def to_par(self, p, chain):
    if "red_noise_log10_A" in p:
        return "TNRedAmp", chain
    if "red_noise_gamma" in p:
        return "TNRedGam", chain
    elif "_log_qp_ratio" in p:
        return "TN_QpRatio", chain
    elif "_log_f0" in p:
        return "TN_QpF0", chain
    elif "_p" in p:
        return "TN_QpPeriod", chain
    elif "_sig" in p:
        return "TN_QpSig", chain
    elif "_lam" in p:
        return "TN_QpLam", chain
    elif "fsin" in p:
        return p,chain
    elif "fcos" in p:
        return p, chain
    else:
        return None


from .chol_red_model import qp_term_cutoff


@function
def powerlaw_qp(f, log10_A, gamma, log_qp_ratio, p, sig, lam, fmin, components=2):
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    df2 = np.repeat(df, components)
    red = (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * const.fyr ** (gamma - 3) * f ** (-gamma) * df2

    f0 = 1.0 / p
    logPqp = log_qp_ratio + np.log10(
        (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * const.fyr ** (gamma - 3) * (f0 * 365.25 * const.fyr) ** (
            -gamma) * df2)
    qp = qp_term_cutoff(f / const.fyr, logPqp, f0, sig, lam, fmin)

    return red + qp


@function
def sinusoid(toas,A,B,om):
    return B*np.sin(om*toas)+A*np.cos(om*toas)



