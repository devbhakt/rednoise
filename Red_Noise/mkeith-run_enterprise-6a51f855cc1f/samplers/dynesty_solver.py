import dynesty
import numpy as np
from multiprocessing import Pool
import pickle, os
from dynesty import plotting
import matplotlib.pyplot as plt

name = "DynestySolver"
argdec = "Configure for nested sampling with DyNesty"


def setup_argparse(parser):
    parser.add_argument('--dynesty', action='store_true', help='Use dynesty sampler')
    parser.add_argument('--dynesty-plots', action='store_true', help="make dynesty run plots")
    parser.add_argument('--dynesty-bound-eff', type=float, default=10.0, help="Efficiency to start bounding")
    parser.add_argument('--dynesty-bound',default="multi", help="Bounding method")
    parser.add_argument('--dynesty-sampler',default="auto", help="Sampling method")
    parser.add_argument('--dynesty-bootstrap',default=0,type=int,help="Bootstrap amount")


def activated(args):
    return args.dynesty


def run_solve(args, _pta, outdir):
    global invTs
    global pta
    invTs = []
    pta = _pta

    for p in pta.params:
        invTs.append(p.invT)

    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, "dynesty_results.pkl")

    nlive = args.nlive
    print("Nlive: {} ndim: {}".format(nlive, len(pta.params)))
    if nlive < 2 * len(pta.params):
        nlive = 2 * len(pta.params)
        print("Warning... nlive too small, setting to {}".format(nlive))

    args.burn = 0  ## force no burn in for nested sampling
    if args.sample:
        with Pool(args.nthread) as tpool:
            print("Run Dynesty")
            sampler = dynesty.NestedSampler(get_lnlikelihood, prior_transform, ndim=len(pta.params), nlive=nlive,
                                            pool=tpool, queue_size=args.nthread, sample=args.dynesty_sampler, bound=args.dynesty_bound,
                                            first_update={'min_ncall': 1000, 'min_eff': args.dynesty_bound_eff},bootstrap=args.dynesty_bootstrap)
            sampler.run_nested(maxiter=args.nsample)
            print(sampler.results.summary())
            with open(filename, "wb") as f:
                pickle.dump(sampler.results, f)
            # print(sampler.results)
            return sampler.results
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def get_posteriors(args, results):
    try:
        weights = np.exp(results['logwt'] - results['logz'][-1])
    except:
        weights = results['weights']

    return results.samples, results.logz, weights, results


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
