import glob
import pickle

from astropy.io import fits
import numpy as np
import pylab as pl
from scipy.integrate import cumtrapz
from scipy.stats import poisson

outpath = '/tank/kerrm/fermi_pta/unbinned'

tn_pulsars = ['J2043+1711','J2256-1024']#,'J0614-3329','J0101-6422']

def load_single(fname):
    with open(fname,'r') as f:
        lines = f.readlines()
        if lines[0].strip() == 'Failure':
            return None
        jname,wn,lim1,lim2 = lines[0].strip().split()
        #wn = float(wn)
        #lim1 = float(lim1)
        #lim2 - float(lim2)
        gwb_dom = np.asarray([float(x) for x in lines[1].strip().split()])
        cod = np.asarray([float(x) for x in lines[2].strip().split()])
        cod_marg = np.asarray([float(x) for x in lines[3].strip().split()])
    return jname,wn,lim1,lim2,cod,cod_marg,gwb_dom

def load_unbinned_results_new(index=None):
    #fnames = sorted(glob.glob('%s/*unbinned_results.asc'%outpath))
    if index is None:
        fnames = sorted(glob.glob('ancillary_data/unbinned_results/*unbinned_results.asc'))
    else:
        fnames = sorted(glob.glob('ancillary_data/unbinned_results/*unbinned_results_%.2f.asc'%index))
    fnames = [x for x in fnames if 'J1311-3430' not in x]
    fnames = [x for x in fnames if 'J1555-2908' not in x]
    fnames = [x for x in fnames if 'J2215+5135' not in x]
    fnames = [x for x in fnames if 'J2339-0533' not in x]
    results = []
    for fname in fnames:
        rvals = load_single(fname)
        if rvals is None:
            print('Failure for %s.'%fname)
        else:
            results.append(rvals)
    # strip out some of the other pulsars just to make life easier
    jnames = np.asarray([r[0] for r in results])
    wn = np.asarray([float(r[1]) for r in results])
    lim1 = np.asarray([float(r[2]) for r in results])
    lim2 = np.asarray([float(r[3]) for r in results])
    cods = np.asarray([r[4] for r in results])
    cod_margs = np.asarray([r[5] for r in results])
    gwb_dom = results[0][-1]

    logl2 = np.loadtxt('ancillary_data/lat_msps.asc',usecols=[2]).transpose()
    jnames2 = np.loadtxt('ancillary_data/lat_msps.asc',usecols=[0],dtype=str).transpose()
    logl = np.ravel(np.asarray(
            [logl2[np.flatnonzero(jnames2==jname)] for jname in jnames]))

    return jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs

def print_summary(limsort=True,wnsort=False,maxn=None):
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    idx = np.arange(len(jnames))
    if limsort:
        idx = np.argsort(lim1)
    if wnsort:
        idx = np.argsort(wn)
    def pad(s,n=10):
        return ' '*(n-len(s))+s
    print('JName            LogL       WN     Lim1     Lim2')
    counter = 0
    for i in idx:
        if (maxn is not None) and (counter == maxn):
            break
        counter += 1
        logls = pad('%.2f'%logl[i])
        wns = pad('%.2f'%wn[i],8)
        lim1s = pad('%.2f'%lim1[i],8)
        lim2s = pad('%.2f'%lim2[i],8)
        print('%s %s %s %s %s'%(jnames[i],logls,wns,lim1s,lim2s))

    print(counter)


def diag_plots():
    # plot the pdfs for normal and marginal likelihoods for each pulsar
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    for ijname,jname in enumerate(jnames):
        pl.figure(1); pl.clf()
        pl.semilogx(gwb_dom,cods[ijname],color='C0',ls='-')
        pl.semilogx(gwb_dom,cod_margs[ijname],color='C0',ls='--')
        pl.axis([5e-17,5e-13,-1.2,1.1])
        #pl.legend(ncol=1,loc='upper left',fontsize='small')
        pl.rcParams['xtick.labelsize'] = 'large'
        pl.rcParams['ytick.labelsize'] = 'large'
        #pl.rcParams['xtick.major.size'] = 6
        #pl.rcParams['xtick.major.pad'] = 6
        pl.xlabel('A$_{\mathrm{gwb}}$',size='x-large')
        pl.ylabel('Log Likelihood (Single and Marginal)', size='x-large')
        pl.savefig('%s/%s_diag.png'%(outpath,jname))

def check_start_finish():
    # check start and finish for pulsars for which we have results
    import utils
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    for ijname,jname in enumerate(jnames):
        par = '/home/kerrm/pp/%s/timing/%s_tnfit.par'%(jname,jname)
        start,finish = utils.get_bounds(par)
        print('%s %d %d'%(jname,round(float(start)),round(float(finish))))
        if abs(float(start)-54682) > 2:
            print('START issue')
        if abs(float(finish)-59245) > 2:
            print('FINISH issue')

def get_lim(c,gwb_dom,val=0.95,normalize_pdf=False):
    """ Integrate the tabulated log likelihood (c) evaluated at the A_gwb
    values (gwb_dom) to get a val*100% confidence limit.
    """
    cod = c
    #pdf = np.exp(cod[0]-cod)
    assert(cod[0]==0)
    pdf = np.exp(cod)
    cpdf = np.append(0,cumtrapz(pdf,x=gwb_dom))
    if normalize_pdf:
        pdf *= 1./cpdf[-1]
    # this is equivalent to a prior at 1e-10
    cpdf *= 1./cpdf[-1]
    a = np.searchsorted(cpdf,val)
    if cpdf[a] > val:
        # interpolate linearly in pdf, logarithmically in GWB amp
        frac = (val-cpdf[a-1])/(cpdf[a]-cpdf[a-1])
        gwb_lim = np.exp(frac*np.log(gwb_dom[a]/gwb_dom[a-1])+np.log(gwb_dom[a-1]))
    else:
        gwb_lim = gwb_dom[a]
    return pdf,cpdf,gwb_lim

def combine_lims(wn_lim=500,limit_mode='hybrid',index=None,quiet=False):
    # WN units: mu^2 yr, so 500 <--> 5e-10 s^2 yr
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new(index=index)
    if limit_mode == 'hybrid':
        if not quiet:
            print('Using marginal likelihood for TN pulsars.')
        use_cod = cods.copy()
        for ijname,jname in enumerate(jnames):
            if jname in tn_pulsars:
                use_cod[ijname] = cod_margs[ijname]
        cods = use_cod
    elif limit_mode == 'marg':
        if not quiet:
            print('Using marginal likelihood for all pulsars.')
        cods = cod_margs
    else:
        if not quiet:
            print('Using GWB-only likelihood for all pulsars.')
        pass
    mask_wn = wn < wn_lim
    #lim0 = get_lim(cods[mask_wn].sum(axis=0),gwb_dom)[-1]
    #problems = ['J1810+1744','J0102+4839']
    problems = []
    mask1 = np.asarray([jname not in problems for jname in jnames])
    #print('mask_wn.sum()=',mask_wn.sum())
    #print('mask1.sum()=',mask1.sum())
    #problems += ['J2256-1024','J2241-5236']
    problems = ['J2256-1024']
    mask2 = np.asarray([jname not in problems for jname in jnames])
    #print('mask2.sum()=',mask2.sum())
    problems += ['J2043+1711']
    mask3 = np.asarray([jname not in problems for jname in jnames])
    #print('mask3.sum()=',mask3.sum())
    lim1 = get_lim(cods[mask1&mask_wn].sum(axis=0),gwb_dom)[-1]
    lim2 = get_lim(cods[mask2&mask_wn].sum(axis=0),gwb_dom)[-1]
    lim3 = get_lim(cods[mask3&mask_wn].sum(axis=0),gwb_dom)[-1]
    return lim1,lim2,lim3

def bootstrap_lim(wn_lim=500,use_marg=True,n0=1,n1=3,n2=8):
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    mask = wn < wn_lim
    jnames = jnames[mask]
    if use_marg:
        cods = cod_margs[mask]
    else:
        cods = cods[mask]
    wn = wn[mask]
    lim1 = lim1[mask]
    all_cod = cods.sum(axis=0)
    niter = 10000
    lim0 = get_lim(all_cod,gwb_dom)[-1]
    lims = []
    for i in range(niter):

        poiss_picks = poisson.rvs([n0,n1,n2])

        # pick n0 (with replacement) from 50 < WN < 100
        #m = (wn > 20) & (wn < 100)
        m = lim1 <= 10
        x = (np.random.rand(poiss_picks[0])*m.sum()).astype(int)
        idx0 = np.flatnonzero(m)[x]

        # pick n1 (with replacement) from 100 < WN < 200
        #m = (wn > 100) & (wn < 200)
        m = (lim1 > 10) & (lim1 <= 20)
        x = (np.random.rand(poiss_picks[1])*m.sum()).astype(int)
        idx1 = np.flatnonzero(m)[x]

        # pick n2 (with replacement) from 200 < WN < 500
        m = (lim1 > 20) & (lim1 <= 50)
        x = (np.random.rand(poiss_picks[2])*m.sum()).astype(int)
        idx2 = np.flatnonzero(m)[x]

        new_cod = all_cod + cods[idx0].sum(axis=0)
        new_cod += cods[idx1].sum(axis=0)
        new_cod += cods[idx2].sum(axis=0)

        lim = get_lim(new_cod,gwb_dom)[-1]
        lims.append(lim)

    lims = np.asarray(lims)
    print(lims.mean()/lim0)

    return lim0,np.asarray(lims)

def combine_subs():
    """ Look at subsets for the limit."""
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    assert(wn.max()<500)
    subs = [
        'J1231-1411',
        'J0614-3329',
        'J1959+2048',
        'J0030+0451',
        'J1630+3734',
        'J1614-2230',
        'J1939+2134',
        'J1902-5105',
        'J2302+4442',
        'J0034-0534',
        'J0101-6422',
        'J0102+4839',
        'J0340+4130',
        'J0613-0200',
        'J0740+6620',
        'J1124-3653',
        'J1536-4948',
        'J1625-0021',
        'J1810+1744',
        'J1816+4510',
        'J1858-2216',
        'J2017+0603',
        'J2043+1711',
        'J2214+3000',
        'J2241-5236',
        'J2256-1024']
    subs += ['J0533+6759','J2034+3632','J1514-4946']
    assert(len(set(subs))==29)
    m2 = np.asarray([x in subs[:2] for x in jnames])
    m3 = np.asarray([x in subs[:3] for x in jnames])
    m5 = np.asarray([x in subs[:5] for x in jnames])
    m9 = np.asarray([x in subs[:9] for x in jnames])
    mcommon = np.asarray([x in subs for x in jnames])
    l2 = get_lim(cods[m2].sum(axis=0),gwb_dom)[-1]*1e14
    l3 = get_lim(cods[m3].sum(axis=0),gwb_dom)[-1]*1e14
    l5 = get_lim(cods[m5].sum(axis=0),gwb_dom)[-1]*1e14
    l9 = get_lim(cods[m9].sum(axis=0),gwb_dom)[-1]*1e14
    lcommon = get_lim(cods[mcommon].sum(axis=0),gwb_dom)[-1]*1e14
    lfull = get_lim(cods.sum(axis=0),gwb_dom)[-1]*1e14
    print('RAW limits')
    print('----------')
    print('Best    2: %.2f'%l2)
    print('Best    3: %.2f'%l3)
    print('Best    5: %.2f'%l5)
    print('Best    9: %.2f'%l9)
    print('Common %d: %.2f'%(len(subs),lcommon))
    print('All    %d: %.2f'%(len(jnames),lfull))
    l2 = get_lim(cod_margs[m2].sum(axis=0),gwb_dom)[-1]*1e14
    l3 = get_lim(cod_margs[m3].sum(axis=0),gwb_dom)[-1]*1e14
    l5 = get_lim(cod_margs[m5].sum(axis=0),gwb_dom)[-1]*1e14
    l9 = get_lim(cod_margs[m9].sum(axis=0),gwb_dom)[-1]*1e14
    lcommon = get_lim(cod_margs[mcommon].sum(axis=0),gwb_dom)[-1]*1e14
    lfull = get_lim(cod_margs.sum(axis=0),gwb_dom)[-1]*1e14
    print('\nw/RN limits')
    print('-----------')
    print('Best    2: %.2f'%l2)
    print('Best    3: %.2f'%l3)
    print('Best    5: %.2f'%l5)
    print('Best    9: %.2f'%l9)
    print('Common %d: %.2f'%(len(subs),lcommon))
    print('All    %d: %.2f'%(len(jnames),lfull))

def sorted_subs(n=2,use_marg=False):
    # get the limit using the single-pulsar limits to sort
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    if use_marg:
        a = np.argsort(lim2)
    else:
        a = np.argsort(lim1)
    idx = a[:n]
    print('Limit uses:',jnames[idx])
    if use_marg:
        lim = get_lim(cod_margs[idx].sum(axis=0),gwb_dom)[-1]
    else:
        lim = get_lim(cods[idx].sum(axis=0),gwb_dom)[-1]
    return lim

def cum_lim(wn_lim=500):
    """ Evaluate the cumulative limit as a function of white noise level.
    """
    jnames,wn,logls,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    gwb_lim = lim1
    mask = wn < wn_lim
    cods = cods[mask]
    wn = wn[mask]
    gwb_lim = gwb_lim[mask]
    a1 = np.argsort(wn)
    a2 = np.argsort(gwb_lim)
    cum_lims1 = [get_lim(cods[a1[:i+1]].sum(axis=0),gwb_dom)[-1] for i in range(len(a1))]
    cum_lims2 = [get_lim(cods[a2[:i+1]].sum(axis=0),gwb_dom)[-1] for i in range(len(a1))]
    cum_lims1m = [get_lim(cod_margs[a1[:i+1]].sum(axis=0),gwb_dom)[-1] for i in range(len(a1))]
    cum_lims2m = [get_lim(cod_margs[a2[:i+1]].sum(axis=0),gwb_dom)[-1] for i in range(len(a1))]
    pl.figure(1); pl.clf()
    pl.plot(cum_lims1,marker='o',color='C0',label='sorted by WN')
    pl.plot(cum_lims2,marker='^',color='C1',label='sorted by GW')
    pl.plot(cum_lims1m,marker='o',color='C0',ls='--')
    pl.plot(cum_lims2m,marker='^',color='C1',ls='--')
    pl.xlabel('Pulsars in Limit')
    pl.ylabel('$A_{gwb}$')
    pl.legend()

# plot 1 -- see "write_lat_msps.py" -- WN vs logl
# plot various GWB limits

def gwb_comp_plot(plot_individual=False,opt_ratio=0.95):
    #jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    #l0,l1,l2,l3 = combine_lims2(limit_mode=')
    #print('Fermi limit: ',l2,l3)
    scale = 1e14
    fermi_limit = 1e-14*scale # representative value
    epta_11 =[2011,6.0e-15*scale]
    epta_15 =[2015,3.0e-15*scale]
    ng5 = [2013,7.00e-15*scale]
    ng11 = [2018,1.45e-15*scale]
    ng12_5 = [2020,1.92e-15*scale]
    ipta_dr1 = [2016,1.7e-15*scale]
    shannon_13 = [2013,2.4e-15*scale]
    shannon_15 = [2015,1e-15*scale]
    jenet_06 = [2006,1.1e-14*scale]
    ppta_21 = [2021,2.2e-15*scale]
    pl.rcParams['xtick.labelsize'] = 'large'
    pl.rcParams['ytick.labelsize'] = 'large'
    pl.figure(2); pl.clf()
    ax = pl.gca()
    ax.set_yscale('log')
    ax.plot([2021],[fermi_limit],color='C3',alpha=1.0,marker='*',ms=20,label='Fermi 2021',ls=' ')
    dom = np.arange(2021,2034)
    cod = fermi_limit*(((12.5+dom-2021)*(1./12.5))**(-13./6))
    ax.plot(dom,cod,color='C3',alpha=1.0,marker=None,ms=5,label=None,ls='--',lw=2)
    #ax.plot(dom,cod*opt_ratio,color='C3',alpha=1.0,marker=None,ms=5,label=None,ls='--')
    if plot_individual:
        ax.plot(2021+np.linspace(-0.5,0.5,len(lims)),np.sort(lims),marker='*',ls=' ',color='C3',alpha=0.5,label='Fermi individual')
    ax.plot(*epta_11,label='EPTA 2011',marker='v',color='C1',ms=7,ls=' ')
    ax.plot(*epta_15,label='EPTA 2015',marker='v',color='C1',ms=7,ls=' ')
    ax.plot(*ipta_dr1,label='IPTA DR1',marker='v',color='C2',ms=7,ls=' ')
    ax.plot(*ng5,label='NANOGrav 5.0 yr',marker='v',color='C0',ms=7,ls=' ')
    ax.plot(*ng11,label='NANOGrav 11.0 yr',marker='v',color='C0',ms=7,ls=' ')
    ax.plot(*ng12_5,label='NANOGrav 12.5 yr',marker='o',color='C0',ms=7,ls=' ')
    ax.plot(*jenet_06,label='PPTA 2006',marker='v',color='C4',ms=7,ls=' ')
    ax.plot(*shannon_13,label='PPTA 2013',marker='v',color='C4',ms=7,ls=' ')
    ax.plot(*shannon_15,label='PPTA 2015',marker='v',color='C4',ms=7,ls=' ')
    ax.plot(*ppta_21,label='PPTA 2021',marker='o',color='C4',ms=7,ls=' ')
    #ax.plot('Combined
    if plot_individual:
        ax.axis([2005,2035,8e-16*scale,1e-12*scale])
    else:
        #ax.axis([2005,2035,8e-16,5e-14])
        ax.axis([2005,2035,0,1.6e-14*scale])
        ax.set_yscale('linear')
    ax.set_xlabel('Published Year',size='x-large')
    ax.set_ylabel(r'$A_{\mathrm{gwb}} \times 10^{-14}$',size='x-large')
    if plot_individual:
        pl.legend(loc='upper right',frameon=True)
    else:
        pl.legend(loc='upper center',ncol=3,frameon=True)
    pl.tight_layout()
    pl.savefig('limit_comp_plot.pdf')

def gwb_comp_plot2(plot_individual=False,plot_fermi=True):
    """ Produce "Figure 1" in the main text."""
    #jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    #l0,l1,l2,l3 = combine_lims2(limit_mode=')
    #print('Fermi limit: ',l2,l3)


    #COLOR BLIND PALETTE
    #IPTA,EPTA,PPTA,NG,Fermi
    CB_CC = ['#ff7f00', '#4daf4a', '#984ea3','#a65628','#e41a1c','#34495e']

    scale = 1e14
    fermi_limit = 1e-14*scale # representative value
    epta_11 =[2011+(7-0.5)/12,6.0e-15*scale]
    epta_15 =[2015+(11-0.5)/12,3.0e-15*scale]
    ng5 = [2013+(1-0.5)/12,7.00e-15*scale]
    ng11 = [2018+(5-0.5)/12,1.45e-15*scale]
    ipta_dr1 = [2016+(5-0.5)/12,1.7e-15*scale]
    shannon_13 = [2013 + (10-0.5)/12,2.4e-15*scale]
    shannon_15 = [2015 + (9-0.5)/12,1e-15*scale]
    jenet_06 = [2006 + (12-0.5)/12,1.1e-14*scale]

    ng12_5 = [(2020+(12-0.5)/12,2020+(12-0.5)/12),(1.37e-15*scale,2.67e-15*scale)]
    ppta_21 = [(2021 + (7-0.5)/12,2021 +(7-0.5)/12),
            (1.9e-15*scale,2.6e-15*scale)]
    epta_21 = [(2021+(11-0.5)/12,2021+(11-0.5)/12),
            (2.3e-15*scale,3.7e-15*scale)]
    ipta_21 = [(2022+(3-0.5)/12,2022+(3-0.5)/12),(2.0e-15*scale,3.6e-15*scale)]

    #SETUP PLOT
    fig = pl.figure(figsize=(18,12))
    pl.clf()

    ax = pl.gca()
    try:
        [i.set_linewidth(2) for i in ax.spines.itervalues()]
    except ValueError:
        [i.set_linewidth(2) for i in ax.spines.values()]
    ax.xaxis.set_major_locator(pl.MaxNLocator(4))
    ax.yaxis.set_major_locator(pl.MaxNLocator(2))
    ax.tick_params(axis='x', labelsize=25,direction='in')
    ax.tick_params(axis='y', labelsize=25,direction='in')
    ax.set_xlabel('Year',size=30)
    ax.set_ylabel(r'$A_{\mathrm{gwb}} (\times 10^{-14}$)',size=30)

    arrow_dxdy = [0.0,-0.12]
    arrow_kwargs = dict(width=0.08,head_width=0.3,head_length=0.035,length_includes_head=True)

    #FERMI
    ax.plot([2022.3],[fermi_limit],color=CB_CC[4],alpha=1.0,marker='*',ms=30,label='Fermi-LAT 2021',ls=' ')
    ax.arrow(*([2022.3,fermi_limit]+arrow_dxdy),color=CB_CC[4],**arrow_kwargs)


    #FERMI SCALINGS
    dom = np.linspace(2015,2025,200)
    cod = fermi_limit*(((12.5+dom-2022.3)*(1./12.5))**(-13./6))
    ax.plot(dom,cod,color=CB_CC[4],alpha=0.7,marker=None,ms=5,ls='--',lw=3)
    ys = (dom-dom[0])*1./(dom[-1]-dom[0]) # ranges 0 to 1
    ax.annotate("Time Scaling", (2018,2),fontsize=26,color=CB_CC[4],rotation=-65,alpha=1.0)
    print(ys)

    # conservative
    #ax.plot(dom,cod*(1-ys*0.1),color=CB_CC[4],alpha=1.0,marker=None,ms=5,label="Conservative",ls='-.',lw=2)
    # optimistic
    #ax.plot(dom,cod*(1-ys*0.25),color=CB_CC[4],alpha=1.0,marker=None,ms=5,label="Optimistic",ls='dotted',lw=2)

    #Annotate scaling lines
    #ax.annotate("10%",(2033,0.2),col1 1 or="#95a5a6",fontsize=20)
    #ax.annotate("25%",(2033,0.1),color="#95a5a6",fontsize=20)

    #SINGLE PULSAR LIMITS FERMI UNBINNED
    if plot_individual:
        ax.plot(2021+np.linspace(-0.5,0.5,len(lims)),np.sort(lims),marker='*',ls=' ',color='C3',alpha=0.5,label='Fermi individual')

    #RADIO DETECTIONS
    #ax.plot(*ipta_21,label="IPTA DR2",marker='s',color=CB_CC[0],ms=10,ls='-',alpha=0.8,lw=2)
    ipta_alpha=1.0 #0.8
    arrow_kwargs['alpha'] = ipta_alpha
    ax.plot(*ipta_21,label=None,marker='s',color=CB_CC[0],ms=10,ls='-',alpha=ipta_alpha,lw=2)
    ax.plot(*ipta_dr1,label='International Pulsar Timing Array (IPTA)',marker='s',color=CB_CC[0],ms=18,ls=' ',alpha=ipta_alpha)
    ax.arrow(*(ipta_dr1+arrow_dxdy),color=CB_CC[0],**arrow_kwargs)

    epta_alpha = 1.0 #0.7
    arrow_kwargs['alpha'] = epta_alpha
    ax.plot(*epta_21,label=None,marker='P',color=CB_CC[1],ms=10,ls='-',alpha=epta_alpha,lw=2)
    ax.plot(*epta_15,label=None,marker='P',color=CB_CC[1],ms=18,ls=' ',alpha=epta_alpha)
    ax.arrow(*(epta_15+arrow_dxdy),color=CB_CC[1],**arrow_kwargs)
    ax.plot(*epta_11,label='European Pulsar Timing Array (EPTA)',marker='P',color=CB_CC[1],ms=18,ls=' ',alpha=epta_alpha)
    ax.arrow(*(epta_11+arrow_dxdy),color=CB_CC[1],**arrow_kwargs)

    ng_alpha = 1.0 #0.8
    arrow_kwargs['alpha'] = ng_alpha
    ax.plot(*ng12_5,label=None,marker='o',color=CB_CC[3],ms=10,ls='-',alpha=ng_alpha,lw=2)
    ax.plot(*ng11,label=None,marker='o',color=CB_CC[3],ms=18,ls=' ',alpha=ng_alpha)
    ax.arrow(*(ng11+arrow_dxdy),color=CB_CC[3],**arrow_kwargs)
    ax.plot(*ng5,label='North American Nanohertz Observatory\n for Gravitational Waves (NANOGrav)',marker='o',color=CB_CC[3],ms=18,ls=' ',alpha=ng_alpha)
    ax.arrow(*(ng5+arrow_dxdy),color=CB_CC[3],**arrow_kwargs)

    ppta_alpha=1.0 #0.8
    arrow_kwargs['alpha'] = ppta_alpha
    ax.plot(*ppta_21,label=None,marker='D',color=CB_CC[2],ms=10,ls='-',alpha=ppta_alpha,lw=2)
    ax.plot(*shannon_15,label=None,marker='D',color=CB_CC[2],ms=18,ls=' ',alpha=ppta_alpha)
    ax.arrow(*(shannon_15+arrow_dxdy),color=CB_CC[2],**arrow_kwargs)
    ax.plot(*shannon_13,label=None,marker='D',color=CB_CC[2],ms=18,ls=' ',alpha=ppta_alpha)
    ax.arrow(*(shannon_13+arrow_dxdy),color=CB_CC[2],**arrow_kwargs)
    ax.plot(*jenet_06,label='Parkes Pulsar Timing Array (PPTA)',marker='D',color=CB_CC[2],ms=18,ls=' ',alpha=ppta_alpha)
    ax.arrow(*(jenet_06+arrow_dxdy),color=CB_CC[2],**arrow_kwargs)

    #PLOTTING DETAILS
    if plot_individual:
        ax.axis([2005,2025,8e-16*scale,1e-12*scale])
    else:
        #ax.axis([2005,2035,8e-16,5e-14])
        ax.axis([2005,2025,-0.05,2.5e-14*scale])
        ax.set_yscale('linear')


    if plot_individual:
        pl.legend(loc='upper left',frameon=True)
    else:
        pl.legend(loc='upper left',ncol=1,frameon=True,fontsize=23.5)

    pl.savefig('GWBLimits-updated.pdf',bbox_inches='tight',format='pdf')
    pl.show()

def unbinned_logl_plot(ts_highlight=False,wn_lim=500,target=None,plot_marg=False):
    """ Compare marginalized and normal likelihoods.
    Fig S2 (plot_marg=False) and S3 (plot_marg=True) in SOM."""
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()
    if target is not None:
        mask = jnames == target
    else:
        mask = wn < wn_lim
    jnames = jnames[mask]
    cods = cods[mask]
    cod_margs = cod_margs[mask]
    ts = cods.max(axis=1)
    if ts_highlight:
        mask = ts > 0.1
        jnames = jnames[mask]
        cods = cods[mask]
        cod_margs = cod_margs[mask]
    pl.figure(3); pl.clf()
    ax = pl.gca()
    ax.set_xscale('log')
    ccounter = 0
    for i in range(len(jnames)):
        label = None
        if (lim1[i] < 15) or ((lim1[i]<35) & (ts[i] > 0.1)):
            label = 'PSR '+jnames[i]
            color = 'C%d'%(ccounter%10)
            alpha=0.9
            ccounter += 1
            ls = ['-','--','-.',':'][int(ccounter/10)]
        else:
            color = 'k'
            alpha=0.3
            ls = ':'
        y = cod_margs[i] if plot_marg else cods[i]
        ax.plot(gwb_dom,y,label=label,color=color,ls=ls,lw=2,alpha=alpha)
        #ax.plot(gwb_dom,cod_margs[i],ls='--')
        #pl.plot(gwb_dom,cods[i]-cod_margs[i],color=color,ls=['-','--','-.',':'][int((i%40)/10)],label=jnames[i])
    #pl.legend(ncol=2,loc='upper left',fontsize='small' if len(jnames) > 30 else 'medium')
    ax.legend(ncol=3,loc='upper left',frameon=False,fontsize='medium')
    ax.set_xlabel('A$_{\mathrm{gwb}}$',size='x-large')
    ax.set_ylabel('Log Likelihood', size='x-large')
    #pl.axis([1e-18,5e-13,-1.2,1.1])
    ax.axis([5e-17,5e-13,-1.2,2.1])
    #pl.savefig('bigplot1.pdf')
    pl.show()
    pl.tight_layout()

def single_pulsar_limit_comparison():
    """ Figure 2 of main text."""
    jnames_a = np.loadtxt('ancillary_data/SinglePulsar_182C_Limits.txt',dtype=str,usecols=[0]).transpose()
    ent,tn = np.loadtxt('ancillary_data/SinglePulsar_182C_Limits.txt',usecols=[1,2]).transpose()*1e14
    jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new()

    fig = pl.figure(figsize=(18,12))
    pl.clf()

    ax = pl.gca()

    toggle1 = False
    toggle2 = False


    ent_color = "#2c3e50"
    tn_color = "#f39c12"
    unbinned_color = "#8e44ad"

    count=0

    for ijname,jname in enumerate(jnames):
        m = np.flatnonzero(jnames_a==jname)

        if len(m) == 0:
            print('No match for %s.'%jname)
            if not toggle1:
                count+=1
                ax.plot([lim1[ijname]],1,marker='^',color=unbinned_color,label="Photon-by-photon only",markersize=20,ls=' ',alpha=0.7)
                toggle1 = True
            else:
                count+=1
                ax.plot([lim1[ijname]],1,marker='^',color=unbinned_color,markersize=20,ls=' ',alpha=0.7)
            continue

        """
        if lim1[ijname] > 60:
            print ('TempoNest/Enterprise: {0}'.format(jname))
            if not toggle2:
                count+=1
                ax.plot([59],[ent[m[0]]],marker='o',ls=' ',color=ent_color,markersize=20,alpha=0.7,label="Enterprise")
                ax.plot([59],[tn[m[0]]],marker='*',ls=' ',color=tn_color,markersize=20.5,alpha=0.9,label="TempoNest")
                toggle2 = True
            else:
                count+=1
                ax.plot([59],[ent[m[0]]],marker='o',ls=' ',color=ent_color,markersize=20,alpha=0.7)
                ax.plot([59],[tn[m[0]]],marker='*',ls=' ',color=tn_color,markersize=20.5,alpha=0.9)
        """
        if not toggle2:
            count+=1
            ax.plot([lim1[ijname]],[ent[m[0]]],marker='o',ls=' ',color=ent_color,markersize=20,alpha=0.7,label="Enterprise")
            ax.plot([lim1[ijname]],[tn[m[0]]],marker='*',ls=' ',color=tn_color,markersize=20.5,alpha=0.9,label="TempoNest")
            toggle2=True
        else:
            count+=1
            print (jname,lim1[ijname],ent[m[0]])
            ax.plot([lim1[ijname]],[ent[m[0]]],marker='o',ls=' ',color=ent_color,markersize=20,alpha=0.7)
            ax.plot([lim1[ijname]],[tn[m[0]]],marker='*',ls=' ',color=tn_color,markersize=20.5,alpha=0.9)
            if abs(lim1[ijname]-ent[m[0]]) > 10:
                if jname == "J1858-2216":
                    ax.annotate("{0}".format(jname),(lim1[ijname]-7,ent[m[0]]-3),fontsize=26)
                else:
                    ax.annotate("{0}".format(jname),(lim1[ijname]+2,ent[m[0]]),fontsize=26)


    print (count)
    ax.set_xlabel(r'Photon-by-photon $A_{\mathrm{gwb}} (\times 10^{-14}$)',size=39)
    ax.set_ylabel(r'TOA-based $A_{\mathrm{gwb}} (\times 10^{-14}$)',size=39)
    ax.plot([0,60],[0,60],ls='--',color='#2c3e50',alpha=0.5,lw=2)
    ax.axis([0,120,0,60])
    pl.legend(fontsize=32,loc='upper right')
    try:
        [i.set_linewidth(2) for i in ax.spines.itervalues()]
    except ValueError:
        [i.set_linewidth(2) for i in ax.spines.values()]
    ax.xaxis.set_major_locator(pl.MaxNLocator(4))
    ax.yaxis.set_major_locator(pl.MaxNLocator(4))
    ax.tick_params(axis='x', labelsize=33,direction='in')
    ax.tick_params(axis='y', labelsize=33,direction='in')
    pl.tight_layout()
    pl.savefig("plots/singlepulsarlimits.pdf",format='pdf', bbox_inches='tight')
    pl.savefig("plots/singlepulsarlimits.png",format='png', bbox_inches='tight')
    pl.show()

def plot_agwb_vs_gamma():
    """ Figure 3 in the main text."""
    grid = np.arange(3,22)*(1./3)
    agwbs = [combine_lims(limit_mode='raw',index=idx,quiet=True)[0] for idx in grid]
    pl.rcParams['xtick.labelsize'] = 'large'
    pl.rcParams['ytick.labelsize'] = 'large'
    pl.figure(1); pl.clf()
    ax = pl.gca()
    # convert to GWB index: gamma = alpha*2-3
    gwb_idx = (-grid+3)*0.5
    ax.plot(gwb_idx,agwbs,marker=None,color='k',alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel(r'Spectral Index ($\alpha$)',size='x-large')
    ax.set_ylabel('$A_{\mathrm{gwb}}$',size='x-large')
    # GWB index = 10
    ax.plot([gwb_idx[10]],[agwbs[10]],marker='*',color='C3',ms=15,label='Supermassive Black Holes',ls=' ',alpha=0.8)
    # scale-invariant, radiation-dominated inflation, alpha=-1, idx = 12
    ax.plot([gwb_idx[12]],[agwbs[12]],marker='^',color='C2',ms=12,label='Relic GW (Inflation)',ls=' ',alpha=0.8)
    # Steve Taylor paper on cosmic strings, alpha=-7./6, idx = 13
    ax.plot([gwb_idx[13]],[agwbs[13]],marker='o',color='C0',ms=12,label='Cosmic Strings',ls=' ',alpha=0.8)
    pl.legend(loc='upper left',fontsize='large')
    ax.tick_params(axis='x',direction='in',which='both')
    ax.tick_params(axis='y',direction='in',which='both')
    ax.fill_between(gwb_idx,agwbs,[5e-13]*len(agwbs),color='k',alpha=0.18,hatch='x')
    ax.axis([-2,1.0,5e-16,5e-13])
    ax.text(0.0,1e-15*0.7,'Allowed Region',size='x-large')
    pl.tight_layout()

def plot_wn_vs_logl():
    """ This is the "sample" figure in the supplementary text (Fig S1)."""

    logl = np.loadtxt('ancillary_data/lat_msps.asc',usecols=[2]).transpose()
    wn = np.loadtxt('ancillary_data/lat_msps.asc',usecols=[3]).transpose()*1e12
    jnames = np.loadtxt('ancillary_data/lat_msps.asc',usecols=[0],dtype=str).transpose()
    print('loaded %d pulsars'%(len(jnames)))
    unbinned_mask = (wn<500) & (logl>150)
    unbinned_mask &= jnames != 'J1311-3430'
    unbinned_mask &= jnames != 'J1555-2908'
    unbinned_mask &= jnames != 'J2215+5135'
    unbinned_mask &= jnames != 'J2339-0533'
    toa_mask = unbinned_mask & (logl > 300)
    print(unbinned_mask.sum())
    print(toa_mask.sum())
    unbinned_color = "#8e44ad"
    tn_color = "#f39c12"
    ent_color = "#2c3e50"
    pl.figure(1); pl.clf()
    pl.rcParams['xtick.labelsize'] = 'large'
    pl.rcParams['ytick.labelsize'] = 'large'
    ax = pl.gca()
    m0 = ~unbinned_mask
    ax.loglog(logl[m0],wn[m0],marker='o',ls=' ',color=ent_color,ms=7,alpha=0.5,label='Remaining MSP sample',mec='none')
    m1 = toa_mask
    ax.loglog(logl[m1],wn[m1],marker='*',ls=' ',color=tn_color,ms=13,label='TOA-based analysis',mec='none')
    m2 = unbinned_mask & ~toa_mask
    ax.loglog(logl[m2],wn[m2],marker='^',ls=' ',color=unbinned_color,ms=12,alpha=0.7,label='Photon-by-photon only',mec='none')
    ax.set_xlabel('Log Likelihood',size='x-large')
    ax.set_ylabel('White Noise ($\mathrm{\mu s}^2\,\mathrm{yr}^{-1}$)',size='x-large')
    ymin = ax.axis()[2]
    xmax = ax.axis()[1]
    ax.vlines(150,ymin,500,linestyles='--',alpha=0.5)
    ax.vlines(300,ymin,500,linestyles='--',alpha=0.5)
    ax.hlines(500,150,xmax,linestyles='--',alpha=0.5)
    ax.axis([ax.axis()[0],xmax,ymin,ax.axis()[3]])
    pl.legend(loc='upper right',frameon=False,fontsize='large')
    pl.tight_layout()

