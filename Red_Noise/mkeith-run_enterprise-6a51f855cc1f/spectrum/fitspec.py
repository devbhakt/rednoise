import numpy as np
import dynesty
import scipy.optimize as opt
import model_components.xparameter as parameter
import matplotlib.pyplot as plt
from multiprocessing import Pool
import dynesty.plotting as plotting
from dynesty import utils as dyfunc
import pickle

ppf = (86400.0 * 365.25) ** 2




def fit_red(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd[-8:])
    p0=[np.log10(psd[0]), 4,np.log10(Pw_guess),freqs_days[0]*365.25/4]
    try:
        popt,pcov = opt.curve_fit(lambda *x: np.log(pl_red(*x)),freqs_days,np.log(model_psd+Pw_guess),p0=p0)
    except Exception as e:
        print("Error in curve fit")
        print(e)
        popt=p0

    print(p0)
    print(popt)

    Pyr = parameter.Uniform(popt[0]-3,popt[0]+3)
    alpha = parameter.Uniform(0,8)
    fc=freqs_days[0]*365.25/4 #parameter.Uniform(freqs_days[0]*365.25/5,freqs_days[0]*365.25)
    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)


    def loglike(pars):
        psd_model = pl_red(freqs_days, pars[0],pars[1],pars[2],fc)
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pyr.invT(None,pars[0]),alpha.invT(None,pars[1]),Pw.invT(None,pars[2])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=3,nlive=300)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results

    with open("red.samples", "wb") as f:
        pickle.dump(sampler.results, f)

    labels=["logPyr3","alpha","logPw"]


    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("red.traceplot.pdf")

    fig,ax = plotting.cornerplot(res,labels=labels)
    fig.savefig("red.cornerplot.pdf")

    weights=np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err=(ul68-ll68)/2

    psd_10 = pl_red(np.array([0.1/365.25]), *medians,fc=fc)[0]

    with open("red.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i], err[i]))
            print("{} {} {}".format(v,medians[i], err[i]), file=ofile)
        print("fc {} 0".format(fc))
        print("fc {} 0".format(fc), file=ofile)
        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1], res.logzerr[-1]),file=ofile)

        print("deltaf {} 0".format(freqs_days[0] * 365.25), file=ofile)

    psd_model = pl_red(freqs_days, *medians,fc=fc)
    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("red.spec.pdf")


    return res.logz[-1]

def fit_white(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd)

    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)


    def loglike(pars):
        psd_model = white(freqs_days, pars[0])
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pw.invT(None,pars[0])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=3,nlive=300)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results


    with open("white.samples", "wb") as f:
        pickle.dump(sampler.results, f)
    labels=["logPw"]


    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("white.traceplot.pdf")

    weights = np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err = (ul68 - ll68) / 2

    psd_10 = white(np.array([0.1/365.25]), *medians)[0]
    with open("white.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i],err[i]))
            print("{} {} {}".format(v, medians[i], err[i]), file=ofile)

        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1], res.logzerr[-1]),file=ofile)
        print("deltaf {} 0".format(freqs_days[0] * 365.25), file=ofile)
    psd_model = white(freqs_days, *medians)
    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("white.spec.pdf")

    return res.logz[-1]



def fit_lognorm(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd[-8:])

    Pyr = parameter.Uniform(np.log10(psd[0])-3,np.log10(psd[0])+3)
    mu = parameter.Uniform(-10,-2)
    sig =parameter.Uniform(0.1,10) ## this is a bit silly range I think
    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)


    def loglike(pars):
        #def lognorm(freq_days, logPyr3, mu, sig, logPwhite):
        psd_model = lognorm(freqs_days, *pars)
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pyr.invT(None,pars[0]),mu.invT(None,pars[1]),sig.invT(None,pars[2]),Pw.invT(None,pars[3])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=4,nlive=300)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results

    with open("lognorm.samples", "wb") as f:
        pickle.dump(sampler.results, f)
    labels=["logPyr3","mu","sig","logPw"]

    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("lognorm.traceplot.pdf")
    fig,ax = plotting.cornerplot(res,labels=labels)
    fig.savefig("lognorm.cornerplot.pdf")

    weights = np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err = (ul68 - ll68) / 2

    psd_10 = lognorm(np.concatenate((freqs_days,[0.1/365.25])), *medians)[-1]

    with open("lognorm.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i],err[i]))
            print("{} {} {}".format(v, medians[i], err[i]), file=ofile)
        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1], res.logzerr[-1]),file=ofile)

        print("deltaf {} 0".format(freqs_days[0]*365.25), file=ofile)


    psd_model = lognorm(freqs_days, *medians)

    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("lognorm.spec.pdf")

    return res.logz[-1]

def fit_redpink(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd[-8:])
    p0 = [np.log10(psd[0]), 4, np.log10(Pw_guess), freqs_days[0] * 365.25 / 4]
    try:
        popt, pcov = opt.curve_fit(lambda *x: np.log(pl_red(*x)), freqs_days, np.log(model_psd + Pw_guess), p0=p0)
    except Exception as e:
        print("Error in curve fit")
        print(e)
        popt=p0

    print(p0)
    print(popt)

    Pyr = parameter.Uniform(popt[0]-3,popt[0]+3)
    alpha1 = parameter.Uniform(0,8)
    alpha2 = parameter.Uniform(0,8)
    fc=freqs_days[0]*365.25/4# parameter.Uniform(freqs_days[0]*365.25/5,freqs_days[0]*365.25)
    f_knee= parameter.Uniform(freqs_days[4]*365.25,freqs_days[-1]*365.25)
    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)


    def loglike(pars):
        psd_model = pl_red_pink(freqs_days, pars[0],pars[1],pars[2],pars[3],pars[4],fc=fc)
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pyr.invT(None,pars[0]),alpha1.invT(None,pars[1]),alpha2.invT(None,pars[2]),f_knee.invT(None,pars[3]),Pw.invT(None,pars[4])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=5,nlive=300)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results
    with open("redpink.samples", "wb") as f:
        pickle.dump(sampler.results, f)
    labels = ["logPyr3", "alpha1", "alpha2", "f_knee",  "logPw"]
    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("redpink.traceplot.pdf")
    fig,ax = plotting.cornerplot(res,labels=labels)
    fig.savefig("redpink.cornerplot.pdf")

    weights = np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err = (ul68 - ll68) / 2

    psd_10 = pl_red_pink(np.array([0.1/365.25]), *medians,fc=fc)[0]

    with open("redpink.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i],err[i]))
            print("{} {} {}".format(v, medians[i], err[i]), file=ofile)

        print("fc {} 0".format(fc))
        print("fc {} 0".format(fc), file=ofile)
        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1], res.logzerr[-1]),file=ofile)

        print("deltaf {} 0".format(freqs_days[0]*365.25), file=ofile)

    psd_model = pl_red_pink(freqs_days, *medians,fc=fc)
    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("redpink.spec.pdf")

    return res.logz[-1]

def fit_white_plus_qpnudot(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd[-8:])

    Pyr = parameter.Uniform(np.log10(Pw_guess)-1,np.log10(Pw_guess)+5)
    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)

    f0 = parameter.Uniform(np.log10(freqs_days[3]),np.log10(min(1/10.0,freqs_days[-4])))
    sig=parameter.Uniform(1e-3,0.25)
    lam = parameter.Uniform(0.1,10)

    def loglike(pars):
        #def white_plus_qp_nudot(freq_days, logPyr3, logf0, sig, lam, logPwhite):
        psd_model = white_plus_qp_nudot(freqs_days, pars[0],pars[1],pars[2],pars[3],pars[4])
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pyr.invT(None,pars[0]), f0.invT(None,pars[1]),sig.invT(None,pars[2]),lam.invT(None,pars[3]),Pw.invT(None,pars[4])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=5,nlive=500)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results
    with open("whiteQP.samples", "wb") as f:
        pickle.dump(sampler.results, f)
    labels = ["logPyr3", "f0", "sig", "lam","logPw"]
    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("whiteQP.traceplot.pdf")

    fig,ax = plotting.cornerplot(res,labels=labels)
    fig.savefig("whiteQP.cornerplot.pdf")

    weights = np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err = (ul68 - ll68) / 2

    psd_10 = white_plus_qp_nudot(np.concatenate((freqs_days,[0.1/365.25])), *medians)[-1]
    with open("whiteQP.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i],err[i]))
            print("{} {} {}".format(v, medians[i], err[i]),file=ofile)
        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]),file=ofile)
        print("deltaf {} 0".format(freqs_days[0]*365.25), file=ofile)

    psd_model = white_plus_qp_nudot(freqs_days, *medians)
    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("whiteQP.spec.pdf")

    #spec_var = (psd_model * freqs_days[0] * 365.25) * ppf
    #plt.plot(freqs_days,np.real(complex_spec) / np.sqrt(spec_var))
    #plt.show()

    return res.logz[-1]


def fit_white_plus_qpnudot_lorenz(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd[-8:])

    Pyr = parameter.Uniform(np.log10(Pw_guess)-1,np.log10(Pw_guess)+6)
    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)

    f0 = parameter.Uniform(np.log10(freqs_days[3]),np.log10(min(1/10.0,freqs_days[-4])))
    sig=parameter.Uniform(1e-3,0.25)
    lam = parameter.Uniform(0.1,10)

    def loglike(pars):
        #def white_plus_qp_nudot(freq_days, logPyr3, logf0, sig, lam, logPwhite):
        psd_model = white_plus_qp_nudot_lorenz(freqs_days, pars[0],pars[1],pars[2],pars[3],pars[4])
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pyr.invT(None,pars[0]), f0.invT(None,pars[1]),sig.invT(None,pars[2]),lam.invT(None,pars[3]),Pw.invT(None,pars[4])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=5,nlive=500)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results
    with open("whiteLQP.samples", "wb") as f:
        pickle.dump(sampler.results, f)
    labels = ["logPyr3", "f0", "sig", "lam","logPw"]
    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("whiteLQP.traceplot.pdf")

    fig,ax = plotting.cornerplot(res,labels=labels)
    fig.savefig("whiteLQP.cornerplot.pdf")

    weights = np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err = (ul68 - ll68) / 2

    psd_10 = white_plus_qp_nudot_lorenz(np.concatenate((freqs_days,[0.1/365.25])), *medians)[-1]
    with open("whiteLQP.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i],err[i]))
            print("{} {} {}".format(v, medians[i], err[i]),file=ofile)
        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]),file=ofile)
        print("deltaf {} 0".format(freqs_days[0]*365.25), file=ofile)

    psd_model = white_plus_qp_nudot_lorenz(freqs_days, *medians)
    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("whiteLQP.spec.pdf")

    #spec_var = (psd_model * freqs_days[0] * 365.25) * ppf
    #plt.plot(freqs_days,np.real(complex_spec) / np.sqrt(spec_var))
    #plt.show()

    return res.logz[-1]


def fit_red_plus_qpnudot_cut(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd[-8:])
    p0 = [np.log10(psd[0]), 4, np.log10(Pw_guess), freqs_days[0] * 365.25 / 4]
    try:
        popt, pcov = opt.curve_fit(lambda *x: np.log(pl_red(*x)), freqs_days, np.log(model_psd + Pw_guess), p0=p0)
    except Exception as e:
        print("Error in curve fit")
        print(e)
        popt=p0

    print(p0)
    print(popt)

    Pyr = parameter.Uniform(popt[0]-6,popt[0]+3)
    alpha = parameter.Uniform(0,8)
    fc=freqs_days[0]*365.25/4 #parameter.Uniform(freqs_days[0]*365.25/5,freqs_days[0]*365.25)
    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)

    log_qp_ratio = parameter.Uniform(-0.8,3.5)
    f0 = parameter.Uniform(np.log10(freqs_days[3]),np.log10(min(1/10.0,freqs_days[-4])))
    sig=parameter.Uniform(1e-3,0.25)
    lam = parameter.Uniform(0.01,10)

    def loglike(pars):
        psd_model = pl_plus_qp_nudot_cutoff(freqs_days, pars[0],pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],fc=fc)
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pyr.invT(None,pars[0]),alpha.invT(None,pars[1]),log_qp_ratio.invT(None,pars[2]), \
                         f0.invT(None,pars[3]),sig.invT(None,pars[4]),lam.invT(None,pars[5]),Pw.invT(None,pars[6])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=7,nlive=1000)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results

    with open("redQPc.samples", "wb") as f:
        pickle.dump(sampler.results, f)
    labels = ["logPyr3", "alpha", "log_QP_ratio","f0", "sig", "lam","logPw"]
    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("redQPc.traceplot.pdf")

    fig,ax = plotting.cornerplot(res,labels=labels)
    fig.savefig("redQPc.cornerplot.pdf")

    weights = np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err = (ul68 - ll68) / 2

    psd_10 = pl_plus_qp_nudot_cutoff(np.concatenate((freqs_days,[0.1/365.25])), *medians,fc=fc)[-1]
    with open("redQPc.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i],err[i]))
            print("{} {} {}".format(v, medians[i], err[i]),file=ofile)
        print("fc {} 0".format(fc))
        print("fc {} 0".format(fc), file=ofile)
        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]),file=ofile)
        print("deltaf {} 0".format(freqs_days[0]*365.25), file=ofile)

    psd_model = pl_plus_qp_nudot_cutoff(freqs_days, *medians,fc=fc)
    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("redQPc.spec.pdf")

    #spec_var = (psd_model * freqs_days[0] * 365.25) * ppf
    #plt.plot(freqs_days,np.real(complex_spec) / np.sqrt(spec_var))
    #plt.show()

    return res.logz[-1]


def fit_red_plus_qpnudot_lorenz(freqs_days, psd, complex_spec,model_psd,nthread=4):


    ## Try fitting the model to the model, to just get initial params
    Pw_guess = np.median(psd[-8:])
    p0 = [np.log10(psd[0]), 4, np.log10(Pw_guess), freqs_days[0] * 365.25 / 4]
    popt, pcov = opt.curve_fit(lambda *x: np.log(pl_red(*x)), freqs_days, np.log(model_psd + Pw_guess), p0=p0)
    print(p0)
    print(popt)

    Pyr = parameter.Uniform(popt[0]-6,popt[0]+3)
    alpha = parameter.Uniform(0,8)
    fc=freqs_days[0]*365.25/4 #parameter.Uniform(freqs_days[0]*365.25/5,freqs_days[0]*365.25)
    Pw = parameter.Uniform(np.log10(Pw_guess)-3,np.log10(Pw_guess)+3)

    log_qp_ratio = parameter.Uniform(-0.8,3.5)
    f0 = parameter.Uniform(np.log10(freqs_days[3]),np.log10(min(1/10.0,freqs_days[-4])))
    sig=parameter.Uniform(1e-3,0.25)
    lam = parameter.Uniform(0.1,10)

    def loglike(pars):
        psd_model = pl_plus_qp_nudot_lorenz(freqs_days, pars[0],pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],fc=fc)
        spec_var = (psd_model*freqs_days[0]*365.25)*ppf
        ll = -0.5*np.sum(np.real(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        ll += -0.5*np.sum(np.imag(complex_spec)**2/spec_var) - 0.5*np.sum(np.log(spec_var*2*np.pi))
        if np.isnan(ll):
            ll=-np.inf
        return ll

    def prior_transform(pars):
        return np.array([Pyr.invT(None,pars[0]),alpha.invT(None,pars[1]),log_qp_ratio.invT(None,pars[2]), \
                         f0.invT(None,pars[3]),sig.invT(None,pars[4]),lam.invT(None,pars[5]),Pw.invT(None,pars[6])])

    sampler = dynesty.NestedSampler(loglike,prior_transform,ndim=7,nlive=1000)
    sampler.run_nested()
    print(sampler.results.summary())
    res = sampler.results

    with open("redLQP.samples", "wb") as f:
        pickle.dump(sampler.results, f)
    labels = ["logPyr3", "alpha", "log_QP_ratio","f0", "sig", "lam","logPw"]
    fig,ax = plotting.traceplot(res,labels=labels)
    fig.savefig("redLQP.traceplot.pdf")

    fig,ax = plotting.cornerplot(res,labels=labels)
    fig.savefig("redLQP.cornerplot.pdf")

    weights = np.exp(res.logwt - res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(res.samples, weights)
    medians = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.500, weights)[0]
    ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.841, weights)[0]
    ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, res.samples, 0.159, weights)[0]
    err = (ul68 - ll68) / 2

    psd_10 = pl_plus_qp_nudot_lorenz(np.concatenate((freqs_days,[0.1/365.25])), *medians,fc=fc)[-1]
    with open("redLQP.results","w") as ofile:
        for i,v in enumerate(labels):
            print("{} {} {}".format(v,medians[i],err[i]))
            print("{} {} {}".format(v, medians[i], err[i]),file=ofile)
        print("fc {} 0".format(fc))
        print("fc {} 0".format(fc), file=ofile)
        print("psd10 {} 0".format(psd_10))
        print("psd10 {} 0".format(psd_10), file=ofile)
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]))
        print("logZ {} {}".format(res.logz[-1],res.logzerr[-1]),file=ofile)
        print("deltaf {} 0".format(freqs_days[0]*365.25), file=ofile)

    psd_model = pl_plus_qp_nudot_lorenz(freqs_days, *medians,fc=fc)
    fig=plt.figure()
    plt.loglog(freqs_days,psd)
    plt.loglog(freqs_days,model_psd)
    plt.loglog(freqs_days,psd_model)
    fig.savefig("redLQP.spec.pdf")

    #spec_var = (psd_model * freqs_days[0] * 365.25) * ppf
    #plt.plot(freqs_days,np.real(complex_spec) / np.sqrt(spec_var))
    #plt.show()

    return res.logz[-1]



def pl_red(freqs_days, logPyr3, alpha, logPwhite,fc):
    fc_days = fc / 365.25
    return 10**logPyr3 * np.power(np.power((freqs_days / fc_days), 2) + 1., -alpha / 2.) + 10**logPwhite

def pl_red_pink(freqs_days, logPyr3, alpha1, alpha2, f_knee, logPwhite,fc):
    fc_days = fc / 365.25
    f_knee_days = f_knee / 365.25
    Pyr3 = 10**logPyr3
    return Pyr3 * np.power(np.power((freqs_days / fc_days), 2) + 1., -alpha1 / 2.) + \
           Pyr3 * np.power(np.power((f_knee_days / fc_days), 2) + 1., -alpha1 / 2.) * np.power(
        np.power((freqs_days / f_knee_days), 2) + 1., -alpha2 / 2.) + 10**logPwhite

def lognorm(freq_days, logPyr3, mu,sig, logPwhite):
    f=np.exp(-(np.log(freq_days[0]*365.25) - mu)**2/(2*sig**2)) # we will scale so that P=1 at freq_days[0]
    return (10**logPyr3)*np.exp(-(np.log(freq_days*365.25) - mu)**2/(2*sig**2))/f + 10**logPwhite

def white(freq_days, logPyr3):
    return np.ones_like(freq_days)*10**logPyr3


def lorenzian_term(freq_days, logPyr3, f0, sig, lam):
    ret = np.zeros_like(freq_days)
    A = 10 ** logPyr3

    sigf0 = max(f0 * sig, freq_days[0]/2.0)  # sigma must be at least df/2 to make sense
    for ih in range(1, 10):
        ret += A * np.exp(-(ih - 1) / lam)/ (ih * (1 + ((freq_days - f0 * ih) / (sigf0 * ih)) ** 2))
    return ret

def qp_term(freq_days, logPyr3, f0, sig, lam):
    ret = np.zeros_like(freq_days)
    A=10**logPyr3

    sigf0 = max(f0*sig,0.5*freq_days[0]) # sigma must be at least df to make sense
    for ih in range(1, 10):
        ret += A * np.exp(-(ih - 1) / lam) * np.exp(-(freq_days - f0 * ih) ** 2 / (2 * (ih * sigf0) ** 2))/ih
    return ret

def pl_plus_qp_nudot(freq_days, logPyr3, alpha, log_ratio_qp, logf0, sig,lam, logPwhite,fc):
    f0=10**logf0
    logQP = log_ratio_qp + np.log10(pl_red(f0, logPyr3, alpha,-100,fc)) ## parameteised by the ratio of QP to red noise (-100 implies no white noise here)
    model = ((freq_days/f0)**-4)*qp_term(freq_days,logQP,f0,sig,lam) + pl_red(freq_days, logPyr3, alpha, logPwhite,fc)
    return model

def pl_plus_qp_nudot_lorenz(freq_days, logPyr3, alpha, log_ratio_qp, logf0, sig,lam, logPwhite,fc):
    f0=10**logf0
    logQP = log_ratio_qp + np.log10(pl_red(f0, logPyr3, alpha,-100,fc)) ## parameteised by the ratio of QP to red noise (-100 implies no white noise here)
    model = ((freq_days/f0)**-4)*lorenzian_term(freq_days,logQP,f0,sig,lam) + pl_red(freq_days, logPyr3, alpha, logPwhite,fc)
    return model

def white_plus_qp_nudot(freq_days, logPyr3, logf0, sig,lam, logPwhite):
    f0=10**logf0
    model = white(freq_days,logPwhite)
    model += ((freq_days/f0)**-4)*qp_term(freq_days,logPyr3,f0,sig,lam)
    return model

def white_plus_qp_nudot_lorenz(freq_days, logPyr3, logf0, sig,lam, logPwhite):
    f0=10**logf0
    model = white(freq_days,logPwhite)
    model += ((freq_days/f0)**-4)*lorenzian_term(freq_days,logPyr3,f0,sig,lam)
    return model


def qp_term_cutoff(freq_days, logPyr3, f0, sig, lam):
    ret = np.zeros_like(freq_days)
    A=10**logPyr3

    sigf0 = max(f0*sig,0.5*freq_days[0]) # sigma must be at least df to make sense
    for ih in range(1, 10):
        ret += A * np.exp(-(ih - 1) / lam) * np.exp(-(freq_days - f0 * ih) ** 2 / (2 * (ih * sigf0) ** 2))/ih
    cut=np.ones_like(ret)
    fcut=0.5*(f0-np.sqrt((f0**2 - 16*sigf0**2)))
    cut[freq_days < fcut]=0
    return ret*cut # introduce a hard cut-off at low freq to avoid long tail things

def pl_plus_qp_nudot_cutoff(freq_days, logPyr3, alpha, log_ratio_qp, logf0, sig,lam, logPwhite,fc):
    f0=10**logf0
    logQP = log_ratio_qp + np.log10(pl_red(f0, logPyr3, alpha,-100,fc)) ## parameteised by the ratio of QP to red noise (-100 implies no white noise here)
    model = ((freq_days/f0)**-4)*qp_term_cutoff(freq_days,logQP,f0,sig,lam) + pl_red(freq_days, logPyr3, alpha, logPwhite,fc)
    return model

