#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib

#matplotlib.use('Agg')
import numpy as np

import model_components
import samplers


import argparse

from enterprise.pulsar import Pulsar # At some point remove this dependency

import output_plots

from modelfit import run_modelfit

import scipy.linalg as linalg
import itertools


def run_cholfit(args):
    par = args.par
    tim = args.tim
    print(vars(args))
    print("Read pulsar data")
    psr=Pulsar(par,tim,drop_t2pulsar=False)

    orig_toas=psr.t2pulsar.toas()
    issorted=np.all(orig_toas[:-1] <= orig_toas[1:])

    with open(par) as f:
        parfile = f.readlines()

    ### make model
    model = make_model(args,psr,parfile)

    pars, pmax, ul, ll, means, medians, stds, ul68, ll68 = run_modelfit(args, psr, par, model, parfile)

    model_components.chol_red_model.make_t2_model(psr, args, dict(zip(pars, pmax)))


def make_model(args,psr,parfile):
    cov_funcs=[]
    for model_comp in model_components.cov_models:
        mf = model_comp.setup_model(args,psr,parfile)
        if not mf is None:
            cov_funcs.append(mf)
    res_funcs=[]
    for model_comp in model_components.res_models:
        mf = model_comp.setup_model(args,psr,parfile)
        if not mf is None:
            res_funcs.append(mf(psr))
    return CholModel(psr=psr, cov_funcs=cov_funcs, res_funcs=res_funcs)

class CholModel:
    def __init__(self, psr, cov_funcs, res_funcs):
        self.psr = psr
        self.cov_funcs=cov_funcs
        self.res_funcs=res_funcs

        self.residuals = self.psr.residuals
        self.design_matrix = self.psr.t2pulsar.designmatrix()

    @property
    def params(self):
        ret=[]
        for f in itertools.chain(self.cov_funcs,self.res_funcs):
            for p in f.params:
                if p not in ret:
                    ret.append(p)
        return ret

    @property
    def param_names(self):
        return map(str,self.params)


    def get_lnprior(self,params):
        with np.errstate(divide='ignore'):
            params = params if isinstance(params, dict) else self.map_params(params)
            return sum(p.get_logpdf(params=params) for p in self.params)

    def map_params(self, xs):
        ret = {}
        ct = 0
        for p in self.params:
            n = p.size if p.size else 1
            ret[p.name] = xs[ct : ct + n] if n > 1 else float(xs[ct])
            ct += n
        return ret


    def get_lnlikelihood(self,params):
        design_matrix = self.design_matrix
        ntoa = len(self.residuals)
        CVM=np.zeros((ntoa,ntoa))
        diags=np.zeros(ntoa)

        params = params if isinstance(params, dict) else self.map_params(params)
        for f in self.cov_funcs:
            if callable(f):
                CVM += f(self.psr.toas, params=params)
            get_ndiag = getattr(f, "get_ndiag", None)
            if not get_ndiag is None:
                diags += get_ndiag(params=params)
        CVM += np.diag(diags)
        L = linalg.cholesky(CVM, lower=True)

        log_det_L = np.sum(np.log(np.diag(L)))

        deterministic_model = np.zeros_like(self.residuals)
        for f in self.res_funcs:
            deterministic_model += f.get_delay(params=params)

        residuals = self.residuals - deterministic_model

        white_residuals = linalg.solve_triangular(L, residuals, lower=True)

        white_DM = linalg.solve_triangular(L, design_matrix, lower=True)

        param, p_cvm, chisq = qrfit(white_DM, white_residuals)

        FF = np.dot(white_DM.T, white_DM)
        sign, log_det_FF = np.linalg.slogdet(FF)

        log_det_CVM = log_det_L * 2

        loglike = -0.5 * chisq - 0.5 * log_det_CVM - 0.5 * ntoa * np.log(2 * np.pi) - 0.5*log_det_FF
        return loglike


def qrfit(dm, b):
    """Least Squares fitting useing the QR decomposition"""
    q, r = linalg.qr(dm, mode='economic')
    p = np.dot(q.T, b)

    ri = linalg.inv(r)
    param = np.dot(linalg.inv(r), p)
    newcvm = ri.dot(ri.T)
    model = param.dot(dm.T)
    postfit = b - model
    chisq = np.sum(np.power(postfit, 2))
    newcvm *= (chisq / float(len(b) - len(param)))
    return param, newcvm, chisq


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run 'cholfit' on a single pulsar")
    parser.add_argument('par')
    parser.add_argument('tim')
    parser.add_argument('--outdir', '-o', type=str, help="Output directory for chains etc")
    parser.add_argument('--TEST',action='store_true')

    for model_comp in model_components.cov_models:
        grp = parser.add_argument_group(model_comp.name, model_comp.argdec)
        model_comp.setup_argparse(grp)
    for model_comp in model_components.res_models:
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

    run_cholfit(args)
