import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import dynesty # @todo: remove this dependency on dynesty
from dynesty import plotting # @todo: remove this dependancy on dynesty
from astropy import coordinates as coord
from astropy import units as units
import sys
try:
    import corner
except:
    pass

try:
    import chainconsumer
except:
    pass


from model_components import priors

def setup_argparse(parser):
    parser.add_argument('--white-corner', action='store_true', help='Make the efac/equad corner plots')
    parser.add_argument('--all-corner', '--corner-all', action='store_true', help='Make corner plots with all params')
    parser.add_argument('--plot-chain', action='store_true', help='Make a plot of the chains/posterior samples')
    parser.add_argument('--plot-derived', action='store_true', help='Include derived parameters in corner plots etc.')
    parser.add_argument('--burn', default=0.25, help='Fraction of chain to burn-in (MC only; default=0.25)')
    parser.add_argument('--truth-file', default=None, help='Truths values of parameters; maxlike=maximum likelihood, default=None')
    parser.add_argument('--dump-samples', default=0,type=int, help='Dump N samples of final scaled parameters')
    parser.add_argument('--dump-samples-npy', default=0,type=int, help='Dump N samples of final scaled parameters as a numpy file')
    parser.add_argument('--dump-chain', action='store_true', help='Dump chain after scaling to phyical parameters')
    parser.add_argument('--plotname', default=None, help="Set plot output file name stem")
    parser.add_argument('--smooth',type=float,default=None,help="Factor for smoothing of corner plots (default=don't smooth)")
    parser.add_argument('--skip-plots', action='store_true', help="Skip all plotting (for debugging I guess)")
    parser.add_argument('--plotlibrary',default="corner",help="Which plot library to use for cornerplots (chainconsumer, dynesty, corner)")

def cornerplot(samples, labels, weights=None, truths=None, truth_color='blue',smooth=None,plotlibrary="dynesty"):

    if (plotlibrary=="corner") and 'corner' not in sys.modules:
        plotlibrary="dynesty"

    if (plotlibrary=="chainconsumer") and 'chainconsumer' not in sys.modules:
        plotlibrary="dynesty"

    if plotlibrary=="corner":
        fig = corner.corner(samples,weights=weights, truths=truths,truth_color=truth_color,labels=labels, plot_datapoints=False,smooth=smooth)
    elif plotlibrary=="chainconsumer":
        cc = chainconsumer.ChainConsumer()
        parameters = [p.replace("_", "\_") for p in labels]

        cc.add_chain(samples, parameters=parameters, weights=weights)
        cc.configure(diagonal_tick_labels=False, tick_font_size=6, label_font_size=6, max_ticks=6,
                     colors="#455A64", shade=True, shade_alpha=0.2, bar_shade=True,
                     sigmas=[0, 1, 2, 3], sigma2d=True,smooth=smooth)
        cc.configure_truth(color=truth_color, ls=":", alpha=0.8)
        fig = cc.plotter.plot(figsize='grow', truth=truths)
    else:
        if weights is None:
            weights = np.ones_like(samples[:, 0])
        res = {'samples': samples, 'weights': weights, 'logwt': np.log(weights), 'logz': [0]}
        fig, ax = plotting.cornerplot(res, truths=truths, labels=labels, truth_color=truth_color,smooth=smooth)
        for aa in ax:
            for a in aa:
                if len(a.xaxis.get_label().get_text()) > 0:
                    a.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
                if len(a.yaxis.get_label().get_text()) > 0:
                    a.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
        fig.patch.set_facecolor('white')

    return fig


def make_plots(args, psr, samples, loglike, weights, pars, derived,derived_names,raw_results=None):

    burn = int(len(samples) * args.burn)
    if args.plotname is None:
        plotname=psr.name
    else:
        plotname=psr.name+'_'+args.plotname

    if args.dump_chain:
        if weights is None:
            dumpchain = samples[burn:,:]
        else:
            dumpchain = np.concatenate((samples[burn:,:], weights[burn:,np.newaxis]),axis=1)
        np.savetxt("{}_chain.txt.gz".format(plotname),dumpchain)
        with open("{}_chain_info.txt".format(plotname),"w") as outf:
            for p in pars:
                outf.write("{}\n".format(p))
            if not weights is None:
                outf.write("SAMPLE_WEIGHT\n")

    if args.skip_plots:
        return



    print("Making plots...")
    if args.dynesty and args.dynesty_plots:
        try:
            fig,ax = plotting.runplot(raw_results)
            fig.savefig("{}.runplot.pdf".format(plotname))
        except Exception as e:
            print(e)

        try:
            fig,ax = plotting.traceplot(raw_results, labels=pars)
            fig.savefig("%s.traceplot.pdf" % plotname)
        except Exception as e:
            print(e)

    if args.plot_derived and len(derived_names)>0:
        derived_names_x = []
        for n in derived_names:
            derived_names_x.append("*{}".format(n))
        pars = np.concatenate((pars,derived_names_x))
        samples = np.concatenate((samples,derived),axis=1)

    i_ef = np.array([i for i, v in enumerate(pars) if 'TNEF' in v], dtype=np.int)
    i_eq = np.array([i for i, v in enumerate(pars) if 'TNEQ' in v], dtype=np.int)

    m_ef = np.zeros(len(pars), dtype=np.bool)
    m_ef[i_ef] = True
    m_eq = np.zeros(len(pars), dtype=np.bool)
    m_eq[i_eq] = True

    n_w = np.logical_not(np.logical_or(m_ef, m_eq))
    imax = np.argmax(loglike)
    pmax = samples[imax, :]

    plt.rcParams.update({'font.size': 8})


    # Plot posterior if requested.
    if args.plot_chain:
        print("Plot Chain")
        x = np.arange(samples.shape[0])
        with PdfPages("%s.chain.pdf" % plotname) as pdf:
            for i in range(len(pars)):

                #plt.rc('text',usetex=False) # Test #
                fig = plt.figure(figsize=(16, 8))
                plt.plot(x[:burn], samples[:burn, i], '.', color='gray')
                plt.plot(x[burn:], samples[burn:, i], '.', color='k')
                plt.title("%s" % pars[i])
                plt.ylabel("%s" % pars[i])
                pdf.savefig()
                plt.close()

    print(f"Plot Cornerplot (use {args.plotlibrary})")
    if args.truth_file is None:
        fig = cornerplot(samples[burn:, n_w], labels=pars[n_w], weights=weights,smooth=args.smooth,plotlibrary=args.plotlibrary)
    elif args.truth_file == "maxlike":
        fig = cornerplot(samples[burn:, n_w], labels=pars[n_w], weights=weights, truths=pmax[n_w],smooth=args.smooth,plotlibrary=args.plotlibrary)
        args.truth_file = None ## This logic needs fixing!
    else:
        d = {}
        with open(args.truth_file) as f:
            for line in f:
                key, val = line.rsplit(maxsplit=1)
                d[key] = float(val)
        fiducial= [None]*len(pars[n_w])
        for i in range(len(pars[n_w])):
            if pars[n_w][i].strip("*") in d: # strip * from derived parameters
                fiducial[i] = d[pars[n_w][i].strip("*")]
        print("Parameters:", pars[n_w])
        print("Reference values:", fiducial)
        fig = cornerplot(samples[burn:, n_w], labels=pars[n_w], weights=weights, truths=fiducial, truth_color='r',smooth=args.smooth,plotlibrary=args.plotlibrary)

    fig.savefig("%s.corner.pdf" % plotname)

    print("Plot extra cornerplots")
    # Make optional corner plots
    if args.all_corner:
        if args.truth_file:
            d = {}
            with open(args.truth_file) as f:
                for line in f:
                    key, val = line.rsplit(maxsplit=1)
                    d[key] = float(val)
            fiducial= [None]*len(pars)
            for i in range(len(pars)):
                if pars[i].strip("*") in d: # strip * from derived parameters
                    fiducial[i] = d[pars[i].strip("*")]
            print("Parameters:", pars)
            print("Reference values:", fiducial)
            fig = cornerplot(samples[burn:, :], labels=pars, weights=weights, truths=fiducial, truth_color='r',smooth=args.smooth,plotlibrary=args.plotlibrary)
        else:
            fig = cornerplot(samples[burn:, :], labels=pars, weights=weights,truths=pmax,smooth=args.smooth,plotlibrary=args.plotlibrary)
            fig.savefig("%s.corner_all.pdf" % plotname)

    if args.white_corner:
        if args.truth_file:
            d = {}
            with open(args.truth_file) as f:
                for line in f:
                    key, val = line.rsplit(maxsplit=1)
                    d[key] = float(val)

            fiducial_ef= [None]*len(pars[m_ef])
            for i in range(len(pars[m_ef])):
                if pars[m_ef][i].strip("*") in d: # strip * from derived parameters
                    fiducial_ef[i] = d[pars[m_ef][i].strip("*")]
            print("Parameters EFAC:", pars[m_ef])
            print("Reference values:", fiducial_ef)
            fig = cornerplot(samples[burn:, m_ef], labels=pars[m_ef], weights=weights, truths=fiducial_ef, truth_color='r',smooth=args.smooth,plotlibrary=args.plotlibrary)
            fig.savefig("%s.corner_efac.pdf" % plotname)

            fiducial_eq= [None]*len(pars[m_eq])
            for i in range(len(pars[m_eq])):
                if pars[m_eq][i].strip("*") in d: # strip * from derived parameters
                    fiducial_eq[i] = d[pars[m_eq][i].strip("*")]
            print("Parameters EQUAD:", pars[m_eq])
            print("Reference values:", fiducial_eq)
            fig = cornerplot(samples[burn:, m_eq], labels=pars[m_eq], weights=weights, truths=fiducial_eq, truth_color='r',smooth=args.smooth,plotlibrary=args.plotlibrary)
            fig.savefig("%s.corner_equad.pdf" % plotname)
        else:
            fig = cornerplot(samples[burn:, m_ef], labels=pars[m_ef], weights=weights,truths=pmax[m_ef],smooth=args.smooth,plotlibrary=args.plotlibrary)
            fig.savefig("%s.corner_efac.pdf" % plotname)
            fig = cornerplot(samples[burn:, m_eq], labels=pars[m_eq], weights=weights,truths=pmax[m_eq],smooth=args.smooth,plotlibrary=args.plotlibrary)
            fig.savefig("%s.corner_equad.pdf" % plotname)

    print("Plots saved")


def posterior_fractions(pname, mean, sig, priors, gi=0, prt=True):
    if priors[gi][-1] == priors[gi][0]:
        return
    else:
        fracmin = (mean - priors[gi][0])/(priors[gi][-1] - priors[gi][0])
        fracmax = (priors[gi][-1] - mean)/(priors[gi][-1] - priors[gi][0])
        fracsig = sig/(priors[gi][-1] - priors[gi][0])
    if prt is True:
        print("Posterior mean's positional fraction in prior width for {}:".format(pname), fracmin, fracmax)
        print("Posterior sigma as a fraction of prior width for {}:".format(pname), fracsig)
        print("")


def write_results(args, psr, par, parfile, samples, loglike, weights, pars, derived,derived_names):
    burn = int(len(samples) * args.burn)


    if args.dump_samples_npy > 0:
        print("Dump numpy samples...")
        ## Make a combines samples array
        if len(derived) > 0:
            equal_samples = np.concatenate((samples[burn:, :len(pars)],derived[burn:,:]),axis=1)
        else:
            equal_samples = samples[burn:, :len(pars)]
        ## We might need to re-weight
        if not weights is None:
            equal_samples = dynesty.utils.resample_equal(equal_samples, weights)

        idx=np.arange(len(equal_samples))
        np.random.choice(idx, args.dump_samples, replace=False)
        np.savez_compressed(par+".samples.npz", pars=np.concatenate((pars,derived_names)), samples=equal_samples[idx,:])


    if args.dump_samples > 0:
        print("Dump samples...")
        with open(par+".samples","w") as outf:
            for p in pars:
                outf.write("{} ".format(p.replace(" ","_")))
            outf.write("\n")
            if weights is None:
                equal_samples = samples[burn:, :len(pars)]
            else:
                equal_samples = dynesty.utils.resample_equal(samples[burn:, :len(pars)], weights)

            idx = np.arange(len(equal_samples))
            for i in np.random.choice(idx,args.dump_samples,replace=False):
                s = equal_samples[i]
                for p in s:
                    outf.write("{} ".format(p))
                outf.write("\n")

    outpar = par + ".post"
    resfname = par + ".results"
    with open(outpar, "w") as outf:
        with open(resfname, "w") as resf:
            insfracmin=insfracmax=insfracsig=taufracmin=taufracmax=taufracsig=None
            s = ("{:20s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}".format("#param", " max-like", " mean",
                                                                                         " std", "2.5%", " 15.9%", " 50%",
                                                                                         " 84.1%", " 97.5%"))
            resf.write(s + "\n")
            print(s)

            imax = np.argmax(loglike)
            pmax = samples[imax, :len(pars)]

            ul68,ll68,ul,ll,ulm,llm,means,stds,medians=get_stats(samples[burn:, :len(pars)], weights)

            for p, v, mean, median, sig, um, lm, u, l, u68, l68 in zip(pars, pmax, means, medians, stds, ulm, llm, ul, ll, ul68, ll68):
                s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g}".format(p, v, mean, sig, l, l68, median, u68, u)
                resf.write(s + "\n")
                print(s)
                if p.startswith("GL"):
                    gi = int(p.split('_')[-1]) - 1
                if p.startswith("GLF0(ins"):
                    posterior_fractions(p, mean, sig, priors.glf0i_range, gi=gi)
                elif p.startswith("GLF0(T="):
                    posterior_fractions(p, mean, sig, priors.glf0t_range, gi=gi)
                #elif p.startswith("GLF0_"):
                #    posterior_fractions(p, mean, sig, priors.glf0_range, gi=gi)
                #elif p.startswith("GLF1_"):
                #    posterior_fractions(p, mean, sig, priors.glf1_range, gi=gi)
                elif p.startswith("GLF2_"):
                    posterior_fractions(p, mean, sig, priors.glf2_range, gi=gi)
                elif p.startswith("GLF0D_"):
                    posterior_fractions(p, mean, sig, priors.glf0d_range, gi=gi)
                elif p.startswith("GLTD_"):
                    posterior_fractions(p, mean, sig, priors.gltd_range, gi=gi)
                elif p.startswith("GLF0D2_"):
                    posterior_fractions(p, mean, sig, priors.glf0d2_range, gi=gi)
                elif p.startswith("GLTD2_"):
                    posterior_fractions(p, mean, sig, priors.gltd2_range, gi=gi)
                elif p.startswith("GLF0D3_"):
                    posterior_fractions(p, mean, sig, priors.glf0d3_range, gi=gi)
                elif p.startswith("GLTD3_"):
                    posterior_fractions(p, mean, sig, priors.gltd3_range, gi=gi)

            for line in parfile:
                e = line.split()
                if e[0] in pars or e[0] in derived_names \
                        or (e[0]=="RAJ" and "dRAJ" in pars) \
                        or (e[0]=="DECJ" and "dDECJ" in pars):
                    outf.write("#"+line)
                else:
                    outf.write(line)
            for p,v in zip(pars,pmax):
                ## Still need a couple of special cases...
                if p=="dRAJ":
                    p="RAJ"
                    new_ra = psr.coord.ra + v*units.arcsecond
                    v=new_ra.to_string(unit=units.hourangle,sep=":",precision=8)
                if p == "dDECJ":
                    p = "DECJ"
                    new_dec = psr.coord.dec + v*units.arcsecond
                    v = new_dec.to_string(unit=units.degree, sep=":",precision=8)
                if p=="RAJ_rad":
                    p="RAJ"
                    v=coord.Angle(v*units.rad).to_string(unit=units.hourangle, sep=":",precision=8)
                if p=="DECJ_rad":
                    p="DECJ"
                    v=coord.Angle(v*units.rad).to_string(unit=units.degree, sep=":",precision=8)

                outf.write("%s %s\n" % (p, v))

            if len(derived) > 0:
                print("")
                print("Derived Params")
                print("")

                dimax = np.argmax(loglike)
                dpmax = derived[dimax, :]
                dul68,dll68,dul,dll,dulm,dllm,dmeans,dstds,dmedians=get_stats(derived[burn:, :], weights)

                for p, v, mean, median, sig, um, lm, u, l, u68, l68 in zip(derived_names, dpmax, dmeans, dmedians, dstds, dulm, dllm, dul, dll, dul68, dll68):
                    s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g}".format(p, v, mean, sig, l, l68, median, u68, u)
                    resf.write(s + "\n")
                    print(s)
                for p,v in zip(derived_names,dpmax):
                    outf.write("%s %s\n" % (p, v))

            if all(p is not None for p in [insfracmin, insfracmax, insfracsig]):
                print("Posterior mean as a fraction of prior width for GLF0(instant):", insfracmin, insfracmax)
                print("Posterior sigma as a fraction of prior width for GLF0(instant):", insfracsig)
            if all(p is not None for p in [taufracmin, taufracmax, taufracsig]):
                print("Posterior mean as a fraction of prior width for GLF0(T=taug):", taufracmin, taufracmax)
                print("Posterior sigma as a fraction of prior width for GLF0(T=taug):", taufracsig)

    return pars, pmax, ul, ll, means, medians, stds, ul68, ll68


def get_stats(samps,weights):
    if weights is None:
        ul68 = np.percentile(samps, 84.1, axis=0)
        ll68 = np.percentile(samps, 15.9, axis=0)
        ul = np.percentile(samps, 97.5, axis=0)
        ll = np.percentile(samps, 2.5, axis=0)
        ulm = np.max(samps, axis=0)
        llm = np.min(samps, axis=0)
        means = np.mean(samps, axis=0)
        stds = np.std(samps, axis=0)
        medians = np.median(samps, axis=0)
    else:
        ## use dynesty to resample
        ul68 = np.apply_along_axis(dynesty.utils.quantile, 0, samps, 0.841, weights)[0]
        ll68 = np.apply_along_axis(dynesty.utils.quantile, 0, samps, 0.159, weights)[0]
        ul = np.apply_along_axis(dynesty.utils.quantile, 0, samps, 0.975, weights)[0]
        ll = np.apply_along_axis(dynesty.utils.quantile, 0, samps, 0.025, weights)[0]
        ulm = np.max(samps, axis=0)
        llm = np.min(samps, axis=0)
        medians = np.apply_along_axis(dynesty.utils.quantile, 0, samps, 0.500, weights)[0]
        means, covs = dynesty.utils.mean_and_cov(samps, weights=weights)
        stds = np.sqrt(np.diag(covs))

    return ul68,ll68,ul,ll,ulm,llm,means,stds,medians
