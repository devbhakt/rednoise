import numpy as np
from . import xparameter as parameter

from enterprise.signals import signal_base

import matplotlib.pyplot as plt

name = "RedNoise"
argdec = "RedNoise Parameters. Note that the red noise model is default ENABLED"


def setup_argparse(parser):
    parser.add_argument('--no-red-noise', dest='red', default=True, action='store_false',
                        help='Disable Power Law Red Noise search')
    parser.add_argument('--Ared-max', '-A', type=float, default=-12, help='Max log10A_Red')
    parser.add_argument('--Ared-min', type=float, default=-18, help='Min log10A_Red')
    parser.add_argument('--red-alpha-max', type=float, default=8, help='Max alpha red')
    parser.add_argument('--red-alpha-min', type=float, default=0, help='Min alpha red')
    parser.add_argument('--red-prior-log', action='store_true',
                        help='Use uniform prior in log space for red noise amplitude')
    parser.add_argument('--red-fc', "--fc", type=float, default=0.01, help='corner frequency (default=0.01 per yr)')
    parser.add_argument('--red-fref', "--fref", type=float, default=1.0, help='Reference frequency for A')
    parser.add_argument('--qp', action='store_true', help='Use QP nudot model')
#    parser.add_argument('--qp-f0-min', type=float, default=None)
#    parser.add_argument('--qp-f0-max', type=float, default=None)

    parser.add_argument('--qp-p-min-np', type=float, default=4.0)
    parser.add_argument('--qp-p-min', type=float, default=None)
    parser.add_argument('--qp-p-max', type=float, default=None)
    parser.add_argument('--qp-ratio-max', type=float, default=3.5)
    parser.add_argument('--qp-sigma-max', type=float, default=0.2)
    parser.add_argument('--t2model', type=str, default=None)
    parser.add_argument("--matern32", action='store_true', help="Use a matern 3/2 kernel in time domain")
    parser.add_argument("--matern52", action='store_true', help="Use a matern 5/2 kernel in time domain")
    parser.add_argument("--gp-length-max", type=float, default=3, help="Max log-length scale for time-domain kernels")
    parser.add_argument("--gp-length-min", type=float, default=1, help="Max log-length scale for time-domain kernels")


def setup_model(args, psr, parfile):
    td = setup_timedomain(args, psr, parfile)
    if td is None and args.red:
        # red noise (powerlaw with nC frequencies)
        if args.red_prior_log:
            log10_A = parameter.Uniform(args.Ared_min, args.Ared_max, to_par=to_par)
        else:
            log10_A = parameter.LinearExp(args.Ared_min, args.Ared_max, to_par=to_par)
        alpha = parameter.Uniform(args.red_alpha_min, args.red_alpha_max, to_par=to_par)

        parfile.append("T2Chol_RedFc {}\n".format(args.red_fc))
        parfile.append("T2Chol_RedFref {}\n".format(args.red_fref))

        if args.qp:
            tspan_days = (np.amax(psr.toas) - np.amin(psr.toas)) / 86400.0
            avg_cadence = tspan_days / len(psr.toas)
            #qp_f0_min = np.log10(2.0 / tspan_days) if args.qp_f0_min is None else args.qp_f0_min
            #qp_f0_max = np.log10(min(0.5 / avg_cadence, 1 / 20.0)) if args.qp_f0_max is None else args.qp_f0_max

            qp_p_max = tspan_days/args.qp_p_min_np if args.qp_p_max is None else args.qp_p_max
            qp_p_min = max(2*avg_cadence,  20.0) if args.qp_p_min is None else args.qp_p_min

            log_qp_ratio = parameter.Uniform(-2, args.qp_ratio_max, to_par=to_par)
            #log_f0 = parameter.Uniform(qp_f0_min, qp_f0_max, to_par=to_par)
            qp_period = parameter.Uniform(qp_p_min, qp_p_max, to_par=to_par)
            sig = parameter.Uniform(1e-3, args.qp_sigma_max, to_par=to_par)
            lam = parameter.Uniform(0.01, 10, to_par=to_par)

            df = 86400.0 / (np.amax(psr.toas) - np.amin(psr.toas))
            return generate_covariance_matrix_powerlaw_qp(log10_P=log10_A, alpha=alpha,
                                                          log_qp_ratio=log_qp_ratio, p=qp_period, sig=sig, lam=lam,
                                                          fc=parameter.Constant(args.red_fc),
                                                          fref=parameter.Constant(args.red_fref),
                                                          df=parameter.Constant(df))("Red")
        else:
            return generate_covariance_matrix_powerlaw(log10_P=log10_A, alpha=alpha,
                                                       fc=parameter.Constant(args.red_fc),
                                                       fref=parameter.Constant(args.red_fref))("Red")
    else:
        return td


def timedomain(args):
    return args.matern32 or args.matern52


def setup_timedomain(args, psr, parfile):
    if timedomain(args):
        log_length_scale = parameter.Uniform(args.gp_length_min, args.gp_length_max)
        if args.red_prior_log:
            log10_A = parameter.Uniform(args.Ared_min, args.Ared_max, to_par=to_par)
        else:
            log10_A = parameter.LinearExp(args.Ared_min, args.Ared_max, to_par=to_par)
        if args.matern32:
            return generate_covariance_matrix_matern32(log10_A=log10_A, log_length=log_length_scale)("RedT")
    else:
        return None


@signal_base.function
def generate_covariance_matrix_matern32(toas, log10_A, log_length):
    toas = toas / 86400.0  ## this function needs toas in days
    # ndays = np.ceil(np.amax(toas) - np.amin(toas) + 1e-10)
    model_A = 10 ** log10_A
    model_sigma = 10 ** log_length

    ntoa = len(toas)
    x = np.zeros((ntoa, ntoa), dtype=np.float64)
    for itoa in range(ntoa):
        x[itoa] = np.abs(toas - toas[itoa])

    k1 = np.sqrt(5.0) / model_sigma
    k2 = 5.0 / 3.0 / model_sigma ** 2

    C = model_A * (1.0 + k1 * x + k2 * x ** 2) * np.exp(-x * k1)
    return C


def make_t2_model(psr, args, params):
    t2model = args.t2model
    if t2model is None:
        t2model = "{}.model".format(psr.name)

    print("Write out max-likelihood model file to '{}'".format(t2model))
    if args.red:  ## power law red noise
        if timedomain(args):
            A = 10 ** params['T2Chol_RedA']
            sigma = 10 ** params['T2Chol_RedLogLen']

            mm = ""
            if args.matern52:
                mm = "T2Matern2.5"
            if args.matern32:
                mm = "T2Matern1.5"
            modelstr = "{} {} {}".format(mm, A, sigma)
        else:
            fc = args.red_fc
            fref = args.red_fref
            # We need the actual amplitude...
            A = 10 ** params['T2Chol_RedA'] * (fref / fc) ** params['T2Chol_RedAlpha']
            if args.qp:
                ## power law + QP
                modelstr = "T2PowerLaw_QPc {} {} {} {} {} {} {}".format(params['T2Chol_RedAlpha'],
                                                                        A, fc,
                                                                        params['T2Chol_QpRatio'],
                                                                        1.0/params['T2Chol_QpPeriod'],
                                                                        params['T2Chol_QpSig'],
                                                                        params['T2Chol_QpLam'])
            else:
                modelstr = "T2PowerLaw {} {} {}".format(params['T2Chol_RedAlpha'], A, fc)
    with open(t2model, "w") as f:
        f.write("MODEL T2\n")
        f.write("MODEL {}\n".format(modelstr))


def to_par(self, p, chain):
    if "_log10_P" in p:
        return "T2Chol_RedA", chain
    elif "_alpha" in p:
        return "T2Chol_RedAlpha", chain
    elif "_log_qp_ratio" in p:
        return "T2Chol_QpRatio", chain
    elif "_log_f0" in p:
        return "T2Chol_QpF0", chain
    elif "_p" in p:
        return "T2Chol_QpPeriod", chain
    elif "_sig" in p:
        return "T2Chol_QpSig", chain
    elif "_lam" in p:
        return "T2Chol_QpLam", chain
    elif "_log_length" in p:
        return "T2Chol_RedLogLen", chain
    elif "_log10_A" in p:
        return "T2Chol_RedA", chain
    else:
        return None


@signal_base.function
def generate_covariance_matrix_powerlaw(toas, log10_P, alpha, fc, fref):
    return psd2cov(toas, pl_red, log10_P, alpha, fc, fref)


@signal_base.function
def generate_covariance_matrix_powerlaw_qp(toas, log10_P, alpha, log_qp_ratio, p, sig, lam, fc, fref, df):
    log_f0 = np.log10(1.0/p)
    return psd2cov(toas, pl_plus_qp_nudot_cutoff, log10_P, alpha, log_qp_ratio, log_f0, sig, lam, fc, fref, df)


def psd2cov(toas, psd_function, *args, **kwargs):
    toas = toas / 86400.0  ## this function needs toas in days
    ndays = np.ceil(np.amax(toas) - np.amin(toas) + 1e-10)
    npts = 128
    fc = 0.01 if not 'fc' in kwargs else kwargs['fc']
    while npts < (ndays + 1) * 2 or npts < (2 * 365.25 / fc):
        npts *= 2

    freq = np.fft.rfftfreq(npts, 1 / 365.25)
    P = psd_function(freq, *args, **kwargs)


    t_days = np.arange(npts)
    cov_unscaled = np.fft.irfft(P)
    cov = cov_unscaled.real * 365.25 * (86400.0 * 365.25) ** 2

    #if False:
    #    f2, psd = np.loadtxt("t2.psd", usecols=(0, 1), unpack=True)
    #    t2l, t2c = np.loadtxt("t2.cov", usecols=(0, 1), unpack=True)
#
#        c2 = np.fft.irfft(psd).real* 365.25 * (86400.0 * 365.25) ** 2
#
#        c3 = np.fft.irfft(P[:len(f2)]).real* 365.25 * (86400.0 * 365.25) ** 2
#
#        #plt.loglog(t2l,t2c,ls='none',marker='.')
#        #plt.loglog(t_days[t_days <= ndays],cov[t_days <= ndays],ls='none',marker='.')
#        #plt.loglog(np.arange(len(c2)),c2,ls='none',marker='.')
#        #print(f2.shape,psd.shape,P.shape,freq[:len(f2)].shape)
#        #print(f2[-1],freq[len(f2)-1])
#        #print(f2,freq)
#        #for i in range(len(f2)):
#        #    print("{} {} {} {} {} {}".format(i,freq[i],f2[i],P[i],psd[i],P[i]-psd[i]))
#        #plt.plot(f2,psd-P[:len(f2)])
#        #plt.show()
#
#        #cov = c3

    ntoa = len(toas)
    ii = np.zeros((ntoa, ntoa), dtype=np.float64)
    for itoa in range(ntoa):
        ii[itoa] = np.abs(toas - toas[itoa])
    return np.interp(ii, t_days[t_days <= ndays], cov[t_days <= ndays])


def pl_red(freqs, logPyr3, alpha, fc, fref):
    p = np.power(np.power((freqs / fc), 2) + 1., -alpha / 2.)
    p *= 10 ** logPyr3 * (fref / fc) ** alpha
    return p


def pl_plus_qp_nudot_cutoff(freq, logPyr3, alpha, log_ratio_qp, logf0, sig, lam, fc, fref,df):
    f0 = 10 ** logf0
    logQP = log_ratio_qp + np.log10(pl_red(f0 * 365.25, logPyr3, alpha, fc, fref))
    model = qp_term_cutoff(freq, logQP, f0, sig, lam,df) + pl_red(freq, logPyr3, alpha, fc, fref)

    return model


def qp_term_cutoff(freqyr, logPyr3, f0, sig, lam,df):
    freq = freqyr / 365.25  ## we want freq in per day
    ret = np.zeros_like(freq)
    A = 10 ** logPyr3
    sigf0 = max(f0 * sig, 0.5 * df)  # sigma must be at least df to make sense
    for ih in range(1, 11):
        ret += np.exp(-(ih - 1) / lam) * np.exp(-(freq - f0 * ih) ** 2 / (2 * (ih * sigf0) ** 2)) / ih

    fcut = 0.5 * (f0 - np.sqrt((f0 ** 2 - 16 * sigf0 ** 2)))
    with np.errstate(divide='ignore'):
        s = A * ((freq / f0) ** -4)
    s[freq < fcut] = 0  # introduce a hard cut-off at low freq to avoid long tail things
    return s * ret
