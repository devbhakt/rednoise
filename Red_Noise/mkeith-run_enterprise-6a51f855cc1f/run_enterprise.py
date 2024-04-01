#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib


matplotlib.use('Agg')
import numpy as np

import model_components
import samplers


import argparse

from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals
from enterprise.signals import signal_base
import qr_timing_model

import output_plots

from modelfit import run_modelfit


def run_enterprise(args):

    print(vars(args))

    par = args.par
    tim=args.tim

    print("Read pulsar data")
    psr=Pulsar(par,tim,drop_t2pulsar=False)

    orig_toas=psr.t2pulsar.toas()
    issorted=np.all(orig_toas[:-1] <= orig_toas[1:])

    with open(par) as f:
        parfile = f.readlines()



    ## @todo: Add supermodel stuff


    ### make model
    model = make_model(args,psr,parfile)

    pta = signal_base.PTA(model(psr))

    run_modelfit(args, psr, par, pta, parfile)



def make_model(args,psr,parfile):
    if args.qr:
        model = qr_timing_model.QRTimingModel()
    else:
        model = gp_signals.TimingModel(use_svd=args.svd)

    for model_comp in model_components.all:
        m = model_comp.setup_model(args,psr,parfile)
        if m:
            model += m

    return model



if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run 'enterprise' on a single pulsar")
    parser.add_argument('par')
    parser.add_argument('tim')
    parser.add_argument('--outdir', '-o', type=str, help="Output directory for chains etc")

    for model_comp in model_components.all:
        grp = parser.add_argument_group(model_comp.name, model_comp.argdec)
        model_comp.setup_argparse(grp)


    grp = parser.add_argument_group("Sampling options")
    samplers.std_args(grp)
    for sampler in samplers.all:
        grp = parser.add_argument_group(sampler.name, sampler.argdec)
        sampler.setup_argparse(grp)

    grp = parser.add_argument_group("Output options")
    output_plots.setup_argparse(grp)

    args=parser.parse_args()

    run_enterprise(args)
