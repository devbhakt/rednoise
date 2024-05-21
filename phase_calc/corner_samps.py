#%%
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import pint.models as models

#%%
reader = emcee.backends.HDFBackend('J1231_orig_rseed_chains.h5')
reader2 = emcee.backends.HDFBackend('J1231_phase_calc_rseed_chains.h5')

# %%
m = models.get_model('J1231_updated.par')
#m=models.get_model('J0030_updated.par')
fitkeys = m.free_params
fitkeys.append('PHASE')
burnin=250
# %%
def plot_priors(
    model,
    chains1,
    chains2,
    maxpost_fitvals1=None,
    maxpost_fitvals2=None,
    fitvals=None,
    burnin=100,
    bins=100,
    scale=False,
):
    """Plot of priors and the post-MCMC histogrammed samples

    Show binned samples, prior probability distribution and an initial
    gaussian probability distribution plotted with 2 sigma, maximum
    posterior and original fit values marked.

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        The initial timing model for fitting
    chains1 : dict
        Post MCMC integration chains that contain the fitter keys and post
        MCMC samples from the first set of chains.
    chains2 : dict
        Post MCMC integration chains that contain the fitter keys and post
        MCMC samples from the second set of chains.
    maxpost_fitvals : list, optional
        The maximum posterior values returned from MCMC integration for each
        fitter key. Plots a vertical dashed line to denote the maximum
        posterior value in relation to the histogrammed samples. If the
        values are not provided, then the lines are not plotted
    fitvals : list, optional
        The original parameter fit values. Plots vertical dashed lines to
        denote the original parameter fit values in relation to the
        histogrammed samples. If the values are not provided, then the
        lines are not plotted.
    burnin : int
        The number of steps that are the burnin in the MCMC integration
    bins : int
        Number of bins used in the histogram
    scale : bool
        If True, the priors will be scaled to the peak of the histograms.
        If False, the priors will be plotted independent of the histograms.
        In certain cases, such as broad priors, the priors or histograms
        might not be apparent on the same plot due to one being significantly
        larger than the other. The scaling is for visual purposes to clearly
        plot the priors with the samples
    """
    keys = list(chains1.keys())
    values1 = [chains1[k][burnin:].flatten() for k in keys]
    values2 = [chains2[k][burnin:].flatten() for k in keys]

    # priors = []
    x_range = []
    counts1 = []
    counts2 = []

    for i, key in enumerate(keys[:-1]):
        x_range.append(np.linspace(min(values1[i].min(), values2[i].min()), 
                                   max(values1[i].max(), values2[i].max()), num=bins))
        # priors.append(getattr(model, key).prior.pdf(x_range[i]))
        a1, x = np.histogram(values1[i], bins=bins, density=True)
        a2, x = np.histogram(values2[i], bins=bins, density=True)
        counts1.append(a1)
        counts2.append(a2)

    fig, axs = plt.subplots(len(keys), figsize=(8, 11), constrained_layout=True)

    for i, p in enumerate(keys[:-1]):
        axs[i].set_xlabel(
            f"{str(p)}: Exact Calc = "
            + "{:.6e}".format(maxpost_fitvals1[i])
            + " Phase Calc = "
            + "{:.6e}".format(maxpost_fitvals2[i])
            + " ("
            + str(getattr(model, p).units)
            + ")"
        )
        # axs[i].axvline(
            # -2 * values1[i].std(), color="b", linestyle="--", label="2 sigma"
        # )
        # axs[i].axvline(2 * values1[i].std(), color="b", linestyle="--")
        # axs[i].axvline(
            # -2 * values2[i].std(), color="r", linestyle="--", label="2 sigma (Set 2)"
        # )
        # axs[i].axvline(2 * values2[i].std(), color="r", linestyle="--")
        axs[i].hist(
            values1[i], bins=bins, density=True, alpha=0.5, label="Exact Calc"
        )
        axs[i].hist(
            values2[i], bins=bins, density=True, alpha=0.5, label="Phase Calc"
        )
        # if scale:
        #     axs[i].plot(
        #         x_range[i] - values1[i].mean(),
        #         priors[i] * counts1[i].max() / priors[i].max(),
        #         label="Prior Probability",
        #         color="g",
        #     )
        # else:
        #     axs[i].plot(
        #         x_range[i] - values1[i].mean(),
        #         priors[i],
        #         label="Prior Probability",
        #         color="g",
        #     )
        if maxpost_fitvals1 is not None:
            axs[i].axvline(
                maxpost_fitvals1[i],
                color="c",
                linestyle="--",
                label="Maximum Likelihood: Exact Value",
            )
            axs[i].axvline(
                maxpost_fitvals2[i],
                color="r",
                linestyle="--",
                label="Maximum Likelihood: Calc Value",
            )
        if fitvals is not None:
            axs[i].axvline(
                fitvals[i],
                color="m",
                linestyle="--",
                label="Original Parameter Fit Value",
            )
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].set_axis_off()
    axs[-1].legend(handles, labels)

# %%
def reader_to_samps(reader,fitkeys,burnin=0):
    chains = np.transpose(reader.get_chain(),(1,0,2))
    chains = [chains[:, :, ii].T for ii in range(len(fitkeys))]
    samples = np.transpose(reader.get_chain(discard=burnin), (1, 0, 2)).reshape((-1, len(fitkeys)))
    return dict(zip(fitkeys,chains)),samples
# %%
def plot_chains(chain_dict, file=False):
    npts = len(chain_dict)
    fig, axes = plt.subplots(npts, 1, sharex=True, figsize=(8, 9))
    for ii, name in enumerate(chain_dict.keys()):
        axes[ii].plot(chain_dict[name], color="k", alpha=0.3)
        axes[ii].set_ylabel(name)
    axes[npts - 1].set_xlabel("Step Number")
    fig.tight_layout()
    if file:
        fig.savefig(file)
        plt.close()
    else:
        plt.show()
        plt.close()
# %%
def maxpost(reader,chains,fitkeys,burnin=0):
    blobs = reader.get_blobs()
    lnprior_samps = blobs["lnprior"]
    lnlikelihood_samps = blobs["lnlikelihood"]
    lnpost_samps = lnprior_samps + lnlikelihood_samps
    ind = np.unravel_index(np.argmax(lnpost_samps[:][burnin:]), lnpost_samps[:][burnin:].shape)
    maxpost = [chains[ii][burnin:][ind] for ii in fitkeys]
    return maxpost
# %%
chains_orig,samples_orig = reader_to_samps(reader,fitkeys,burnin)
chains_calc,samples_calc = reader_to_samps(reader2,fitkeys,burnin)
# %%
maxpost_orig = maxpost(reader,chains_orig,fitkeys,burnin)
maxpost_calc = maxpost(reader2,chains_calc,fitkeys,burnin)

# %%
import matplotlib.lines as mlines
blue_line = mlines.Line2D([],[],color='blue',label='Exact Calc')
red_line = mlines.Line2D([],[],color='red',label='DM Calc')
figure = corner.corner(samples_orig,bins=50,labels=fitkeys,truths=maxpost_orig,plot_contours=True,color='blue')
corner.corner(samples_calc,bins=50,labels=fitkeys,truths=maxpost_calc,plot_contours=True,color='red',fig=figure)
plt.legend(handles=[blue_line,red_line],bbox_to_anchor=(0.,1.0,1.,.0), loc=4,fontsize=16)
figure.savefig('J1231_rseed_compare_samps_triangle.png')
plt.close()

# %%
orig_fitvals = np.array([getattr(m,p).value for p in fitkeys[:-1]])

# %%
plot_priors(m,chains_orig,chains_calc,maxpost_orig,maxpost_calc,orig_fitvals,burnin)
plt.savefig('J1231_rseed_compare_samps.png')
plt.close()
# %%
