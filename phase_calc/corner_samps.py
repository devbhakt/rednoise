#%%
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import pint.models as models

#%%
reader = emcee.backends.HDFBackend('original_chains.h5')
reader2 = emcee.backends.HDFBackend('phase_calc_chains.h5')

# %%
m = models.get_model('J1231.par')
fitkeys = m.free_params
fitkeys.append('PHASE')
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
chains_orig,samples_orig = reader_to_samps(reader,fitkeys,burnin=500)
chains_calc,samples_calc = reader_to_samps(reader2,fitkeys,burnin=500)
# %%
maxpost_orig = maxpost(reader,chains_orig,fitkeys,burnin=500)
maxpost_calc = maxpost(reader2,chains_calc,fitkeys,burnin=500)

# %%
figure = corner.corner(samples_orig,bins=50,label=fitkeys,truths=maxpost_orig,plot_contours=True,color='blue')
corner.corner(samples_calc,bins=50,label=fitkeys,truths=maxpost_calc,plot_contours=True,color='red',fig=figure)
figure.savefig('compare_samps_triangle.png')
plt.close()
# %%
