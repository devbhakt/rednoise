import numpy as np
import pickle, os
import matplotlib.pyplot as plt
import pymultinest

try:
    from mpi4py import MPI
    import sys
    _have_mpi=True
except:
    pass



name = "MultiNestSolver"
argdec = "Configure for nested sampling with pymultinest"


def setup_argparse(parser):
    parser.add_argument('--multinest', action='store_true', help='Use pymultinest sampler')
    parser.add_argument('--multinest-prefix', default="pmn-", help='Prefix for pymultinest runs')
    parser.add_argument('--multinest-ce-mode',action="store_true", default=False, help='enable constant efficiency mode')
    parser.add_argument('--multinest-disable-is', dest='multinest_is_mode',action="store_false", help='disable importance sampling')
    parser.add_argument('--multinest-eff', type=float, default=0.8, help="Multinest sampling efficiency")


def activated(args):
    return args.multinest


def run_solve(args, _pta, outdir):
    global invTs
    global pta

    main_process=True
    if _have_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        main_process = rank==0

    invTs = []
    pta = _pta
    ndim=len(pta.params)

    for p in pta.params:
        invTs.append(p.invT)

    prefix=os.path.join(outdir,args.multinest_prefix)

    if main_process:
        os.makedirs(outdir, exist_ok=True)

    nlive = args.nlive
    print("Nlive: {} ndim: {}".format(nlive, len(pta.params)))
    if nlive < 2 * len(pta.params):
        nlive = 2 * len(pta.params)
        print("Warning... nlive too small, setting to {}".format(nlive))

    args.burn = 0  ## force no burn in for nested sampling
    if args.sample:
        print("Run Multinest")
        results = pymultinest.solve(get_lnlikelihood, prior_transform, n_dims=ndim, n_live_points=nlive,
                                    outputfiles_basename=prefix,resume=args.cont, verbose=True,
                                    const_efficiency_mode = args.multinest_ce_mode,
                                    importance_nested_sampling=args.multinest_is_mode,
                                    sampling_efficiency=args.multinest_eff)

    res= {'prefix': prefix, 'npar': len(pta.params)}
    if main_process:
        return res
    else:
        sys.exit(0)


def get_posteriors(args, results):
    prefix=results['prefix']
    n_params = results['npar']
    a = pymultinest.Analyzer(n_params=n_params, outputfiles_basename=prefix)
    s = a.get_stats()
    print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
    data = a.get_data()
    weights = a.get_data()[:, 0]
    return data[:, 2:], -0.5*data[:, 1], weights, results


def prior_transform(u):
    global invTs
    """
    Surely there is a faster way... but I'm not sure how (mkeith jan 2021).
    """
    res = np.zeros_like(u)
    for i in range(len(u)):
        res[i] = invTs[i](u[i])
    return res


def get_lnlikelihood(p):
    global pta
    return pta.get_lnlikelihood(p)
