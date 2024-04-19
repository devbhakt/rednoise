import numpy as np
from enterprise.signals import selections, white_signals
from . import xparameter as parameter

name = "WhiteNoise"
argdec = "WhiteNoise Parameters. White noise defaults to Enabled."


def setup_argparse(parser):
    parser.add_argument('--no-white', dest='white', default=True, action='store_false', help='Disable efac and equad')
    parser.add_argument('--jbo', '-j', action='store_true', help='Use -be flag for splitting backends')
    parser.add_argument('--be-flag', '-f', help='Use specified flag for splitting backends')
    parser.add_argument('--white-prior-log', action='store_true', help='Use uniform prior in log space for Equad')
    parser.add_argument('--efac-max', type=float, default=5, help='Max for efac prior')
    parser.add_argument('--efac-min', type=float, default=0.2, help='Min for efac prior')

    parser.add_argument('--equad-max', type=float, default=None, help='Max for equad prior (default based on median error)')
    parser.add_argument('--equad-min', type=float, default=None, help='Min for equad prior (default based on median error)')
    parser.add_argument('--ngecorr', action='store_true', help='Add ECORR for the nanograv backends')
    parser.add_argument('--ecorr', action='store_true', help='Add ECORR for all')

    parser.add_argument('--ecorr-max', type=float, default=-5, help='Max for ecorr prior')
    parser.add_argument('--ecorr-min', type=float, default=-9, help='Min for ecorr prior')


def setup_model(args, psr, parfile):
    selection = selections.Selection(selections.by_backend)
    selflag = '-f'
    if args.jbo:
        selection = selections.Selection(jbbackends)
        selflag = '-be'
    if args.be_flag:
        selflag = "-" + args.be_flag
        selection = selections.Selection(lambda flags: flagbackends(flags, args.be_flag))

    def to_par(self, p, chain):
        if "efac" in p:
            flag = p.split('_', 1)[1]
            flag = flag[:-5]
            return "TNEF %s %s" % (selflag, flag), chain
        if "equad" in p:
            flag = p.split('_', 1)[1]
            flag = flag[:-14]
            return "TNEQ %s %s" % (selflag, flag), chain
        if "ecorr" in p:
            flag=p.split('_',1)[1]
            flag=flag[:-12]
            return "TNECORR -be %s"%(flag), np.power(10,chain)*1e6 # Convert log-seconds to microseconds
        else:
            return None
    s = selection(psr)

    if args.white:
        medianlogerr = np.log10(np.median(psr.toaerrs))
        efac = parameter.Uniform(args.efac_min, args.efac_max, to_par=to_par)
        if args.equad_min is None:
            equad_min = medianlogerr - 2
        else:
            equad_min = args.equad_min
        if args.equad_max is None:
            equad_max = medianlogerr + 2
        else:
            equad_max = args.equad_max

        if (args.white_prior_log):
            equad = parameter.Uniform(equad_min,equad_max, to_par=to_par)
        else:
            equad = parameter.LinearExp(equad_min,equad_max, to_par=to_par)
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
        model = ef + eq
    else:
        ef = white_signals.MeasurementNoise(efac=parameter.Constant(1.0), selection=selection)
        model = ef


    if args.ecorr:
        if args.white_prior_log:
            ecorr = parameter.Uniform(args.ecorr_min, args.ecorr_max,to_par=to_par)
        else:
            ecorr = parameter.LinearExp(args.ecorr_min,args.ecorr_max,to_par=to_par)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
        model += ec


    if args.ngecorr:
        ngselection = selections.Selection(selections.nanograv_backends)
        if args.white_prior_log:
            ecorr = parameter.Uniform(-9,-5,to_par=to_par)
        else:
            ecorr = parameter.LinearExp(-9,-5,to_par=to_par)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ngselection)
        model += ec

    return model



def jbbackends(flags):
    backend_flags = flags['be']
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def flagbackends(flags, beflagval):
    backend_flags = flags[beflagval]
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}
