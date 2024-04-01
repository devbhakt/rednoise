import numpy as np
import os
import copy
import emcee
from multiprocessing import Pool



name="EmceeSolver"
argdec="Configure for mcmc with EMCEE"

def setup_argparse(parser):
    parser.add_argument('--emcee', action='store_true', help='Use emcee sampler')


def activated(args):
    return args.emcee

def run_solve(args,pta,outdir):
    filename = os.path.join(outdir, "chain.h5")
    if args.sample:
        x0 = np.hstack([p.sample() for p in pta.params])
        os.makedirs(outdir,exist_ok=True)
        nwalkers = args.nwalkers
        if nwalkers < 2*len(x0):
            nwalkers = 2*len(x0)

        if args.cont:
            scls, offs = np.loadtxt(os.path.join(outdir, "scloff"), unpack=True)
        else:
            vv = [np.hstack([p.sample() for p in pta.params]) for i in range(100)]
            scls = np.std(vv, axis=0)
            offs = np.mean(vv, axis=0)
            np.savetxt(os.path.join(outdir, "scloff"), np.array([scls, offs]).T)
        tpool = Pool(args.nthread, initializer=init, initargs=[pta, offs, scls])
        backend = emcee.backends.HDFBackend(filename)
        if args.cont:
            print("CONTINUE EXISTING CHAIN!")
        else:
            backend.reset(nwalkers,len(x0))
        #moves=[(emcee.moves.StretchMove(),0.4),(emcee.moves.DEMove(), 0.5), (emcee.moves.DESnookerMove(), 0.1),]
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        sampler = emcee.EnsembleSampler(nwalkers,len(x0),log_prob,backend=backend,pool=tpool,moves=moves)
        p0=np.zeros((nwalkers,len(x0)))
        for i in range(nwalkers):
            p0[i] = np.hstack([p.sample() for p in pta.params])
            p0[i] -= offs
            p0[i] /= scls

        N = int(args.nsample)
        print("Launch EMCEE\n\n")

        if args.cont:
            p0=None
        state = sampler.run_mcmc(p0,1)
        for state in sampler.sample(state,iterations=N,progress=True):
            if sampler.iteration % 10:
                continue
            i = np.argmin(sampler.acceptance_fraction)
            if sampler.acceptance_fraction[i] < 0.1:
                state.log_prob[i] -= 1e99

            if sampler.iteration % 100:
                continue
            print(sampler.iteration)
            print("Acceptance rate: {:.2f} {:.2f} {}".format(np.mean(sampler.acceptance_fraction),np.amin(sampler.acceptance_fraction),i))
        tpool.close()
    else:
        scls,offs = np.loadtxt(os.path.join(outdir,"scloff"),unpack=True)
    return (scls,offs,filename)


def get_posteriors(args,info):
    scls,offs,filename = info
    reader = emcee.backends.HDFBackend(filename)

    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    print("tau = {} burn = {} thin = {}".format(tau, burnin, thin))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    samples *= scls
    samples += offs
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

    log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

    return samples, log_prob_samples,None, [log_prior_samples]


def init(_pta,_offs,_scls):
    global pta
    global offs
    global scls
    offs=_offs.copy()
    scls=_scls.copy()
    pta = copy.copy(_pta)
    #print("Init Thread")


def log_prob(p):
    global pta
    global scls
    global offs
    p *= scls
    p += offs
    lp=pta.get_lnprior(p)

    if not np.isfinite(lp):
        return -np.inf, -np.inf
    ll=pta.get_lnlikelihood(p)
    if not np.isfinite(ll):
        return -np.inf, -np.inf
    else:
        return ll+lp, lp # get rid of log prior?

