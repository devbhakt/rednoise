
all=[]

try:
    from  . import emcee_solver
    all.append(emcee_solver)
except ImportError:
    print("Cannot import emcee")

try:
    from . import dynesty_solver
    all.append(dynesty_solver)
except ImportError:
    print("Cannot import dynesty")

try:
    from . import zeus_solver
    all.append(zeus_solver)
except ImportError:
    print("Cannot import zeus")

try:
    from . import ptmcmc_solver
    all.append(ptmcmc_solver)
except ImportError:
    print("Cannot import ptmcmc")


try:
    from . import multinest_solver
    all.append(multinest_solver)
except ImportError:
    print("Cannot import pymultinest")



def std_args(parser):
    parser.add_argument('--cont', action='store_true', help='Continue existing run')
    parser.add_argument('--nthread', '-t', type=int, default=1, help="number of threads")
    parser.add_argument('-N','--nsample', type=float, default=1e6, help='(max) number of samples')
    parser.add_argument('-n','--no-sample',dest='sample',default=True,action='store_false', help='Disable the actual sampling...')
    parser.add_argument('--nlive', type=int, default=500, help="Number of live points (nested)")
    parser.add_argument('--nwalkers', type=int, default=0, help="number of walkers (mcmc)")
