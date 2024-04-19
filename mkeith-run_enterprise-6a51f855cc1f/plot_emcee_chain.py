#!/usr/bin/env python

import emcee


import numpy as np

import sys
import matplotlib
matplotlib.use('Agg')

import corner
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



filename=sys.argv[1]


reader = emcee.backends.HDFBackend(filename)
samples = reader.get_chain()


print("samples:",samples.shape)
scls,offs = np.loadtxt("scloff",unpack=True)


pars = np.loadtxt('pars.txt', dtype=np.unicode_)
npar = len(pars)
chain = np.reshape(samples,(-1,npar))
chain*=scls
chain+=offs
x=np.arange(chain.shape[0])
with PdfPages("chain.pdf") as pdf:
    for i in range(len(pars)):
        fig=plt.figure(figsize=(16,8))
        plt.plot(x[:],chain[:,i],'.',color='k')
        plt.title("%s"%pars[i])
        plt.ylabel("%s"%pars[i])
        pdf.savefig()
        plt.close()



tau = reader.get_autocorr_time(tol=0)
print("tau=",tau)
burnin = 0
thin = 10
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
samples*=scls
samples+=offs
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate( (samples, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1)

ndim=len(pars)
labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
labels=list(pars)
labels += ["log prob", "log prior"]

print("TT")
fig = corner.corner(all_samples, labels=labels,smooth=True);
fig.savefig("corner.pdf")

