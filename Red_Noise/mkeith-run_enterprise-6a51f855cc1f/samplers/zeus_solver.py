import numpy as np
import os
import copy
import zeus
from multiprocessing import Pool
import pickle



name="ZeusSolver"
argdec="Configure for mcmc with Zeus"

def setup_argparse(parser):
    parser.add_argument('--zeus', action='store_true', help='Use zeus sampler')


def activated(args):
    return args.zeus

def run_solve(args,pta,outdir):
    filename = os.path.join(outdir, "state")
    x0 = np.hstack([p.sample() for p in pta.params])
    if args.sample:
        nwalkers = args.nwalkers
        if nwalkers < 2*len(x0):
            nwalkers = 2*len(x0)

        tpool = Pool(args.nthread, initializer=init, initargs=[pta])

        sampler = zeus.EnsembleSampler(nwalkers,len(x0),log_prob,pool=tpool)
        p0=np.zeros((nwalkers,len(x0)))
        for i in range(nwalkers):
            p0[i] = np.hstack([p.sample() for p in pta.params])

        N = int(args.nsample)
        print("Launch Zeus\n\n")
        os.makedirs(outdir,exist_ok=True)

        with open(filename,"wb") as outf:
            sampler.run_mcmc(p0,N)
            output = {}
            pickle.dump(sampler.samples,outf)

        tpool.close()
    return (filename,nwalkers,len(x0))


def get_posteriors(args,info):
    filename,nw,np = info
    with open(filename,"rb") as  inf:
        sampler = zeus.EnsembleSampler(nw,np,log_prob)
        sampler.samples=pickle.load(inf)

    tau = zeus.AutoCorrTime(sampler.get_chain())
    burnin = 0#int(2 * np.max(tau))
    thin = 10#int(0.5 * np.min(tau))
    print("tau = {} burn = {} thin = {}".format(tau, burnin, thin))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

    return samples, log_prob_samples,None, [log_prior_samples]


def init(_pta):
    global pta
    pta = copy.copy(_pta)
    #print("Init Thread")


def log_prob(p):
    global pta
    lp=pta.get_lnprior(p)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    ll=pta.get_lnlikelihood(p)
    if not np.isfinite(ll):
        return -np.inf, -np.inf
    else:
        return ll+lp, lp # get rid of log prior?

