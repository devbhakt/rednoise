import numpy as np
import pickle, os
import matplotlib.pyplot as plt
import PTMCMCSampler.PTMCMCSampler as ptmcmc

try:
    from mpi4py import MPI
    import sys

    _have_mpi = True
except:
    pass

name = "PTMCMCSolver"
argdec = "Configure for MCMC sampling with ptmcmcsampler"


def setup_argparse(parser):
    parser.add_argument('--ptmcmc', action='store_true', help='Use ptmcmc sampler')


def activated(args):
    return args.ptmcmc


def run_solve(args, pta, outdir):
    main_process = True
    nprocs = 1
    if _have_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        main_process = rank == 0

    if args.sample:
        cfile = "chain_1.txt"
        if nprocs > 1:
            cfile = "chain_1.0.txt"

        with open(os.path.join(outdir, "cname"), "w") as f:
            f.write(cfile)

        x0 = np.hstack([p.sample() for p in pta.params])
        ndim = len(x0)
        # initial jump covariance matrix
        if os.path.exists(outdir + '/cov.npy') and args.cont:
            cov = np.load(outdir + '/cov.npy')

            # check that the one we load is the same shape as our data
            cov_new = np.diag(np.ones(ndim) * 0.1 ** 2)
            if cov.shape != cov_new.shape:
                msg = 'The covariance matrix (cov.npy) in the output folder is '
                msg += 'the wrong shape for the parameters given. '
                msg += 'Start with a different output directory or '
                msg += 'change resume to False to overwrite the run that exists.'

                raise ValueError(msg)
        else:
            cov = np.diag(np.ones(ndim) * 0.1 ** 2)

        sampler = ptmcmc.PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov=cov, outDir=outdir,
                                   resume=args.cont)
        sampler.sample(x0, int(args.nsample),burn=int(min(10000, 0.1 * args.nsample)))


    else:
        with open(os.path.join(outdir, "cname"))as f:
            cfile = f.readline().strip()

    res = os.path.join(outdir, cfile)
    if main_process:
        return res
    else:
        sys.exit(0)


def get_posteriors(args, info):
    res = info
    data = np.loadtxt(res)

    samples = data[:, :-4]
    log_prob_samples = data[:, -4]

    return samples, log_prob_samples, None, info
