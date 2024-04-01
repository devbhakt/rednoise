# run_enterprise
Single pulsar timing analysis using the [enterprise](https://doi.org/10.5281/zenodo.4059815) framework.
Sorry that there is no documentation! Maybe it will come one day.


You can cite this code and find a perminant URI for versions of it here:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5914351.svg)](https://doi.org/10.5281/zenodo.5914351)

Please cite also [enterprise](https://doi.org/10.5281/zenodo.4059815), [tempo2](https://bitbucket.org/psrsoft/tempo2), and your preferred sampler as appropriate.

## Usage

```
usage: run_enterprise.py [-h] [--outdir OUTDIR] [--no-red-noise]
                         [--Ared-max ARED_MAX] [--Ared-min ARED_MIN]
                         [--red-gamma-max RED_GAMMA_MAX]
                         [--red-gamma-min RED_GAMMA_MIN] [--red-prior-log]
                         [--red-ncoeff RED_NCOEFF] [--tspan-mult TSPAN_MULT]
                         [--qp] [--qp-ratio-max QP_RATIO_MAX]
                         [--qp-sigma-max QP_SIGMA_MAX]
                         [--qp-p-min-np QP_P_MIN_NP] [--qp-p-min QP_P_MIN]
                         [--qp-p-max QP_P_MAX] [--no-white] [--jbo]
                         [--be-flag BE_FLAG] [--white-prior-log]
                         [--efac-max EFAC_MAX] [--efac-min EFAC_MIN]
                         [--ngecorr] [-D] [--Adm-max ADM_MAX]
                         [--Adm-min ADM_MIN] [--dm-ncoeff DM_NCOEFF]
                         [--dm-prior-log] [--dm-tspan-mult DM_TSPAN_MULT]
                         [--dm1 DM1] [--dm2 DM2] [--f2 F2] [--pm] [--px]
                         [--px-range PX_RANGE] [--pm-angle]
                         [--pm-range PM_RANGE] [--pm-ecliptic] [--pos]
                         [--pos-range POS_RANGE] [--legendre]
                         [--leg-df0 LEG_DF0] [--leg-df1 LEG_DF1]
                         [--wrap WRAP [WRAP ...]] [--wrap-range WRAP_RANGE]
                         [--tm-fit-file TM_FIT_FILE] [--glitch-all]
                         [--glitches GLITCHES [GLITCHES ...]]
                         [--glitch-recovery GLITCH_RECOVERY [GLITCH_RECOVERY ...]]
                         [--glitch-double-recovery GLITCH_DOUBLE_RECOVERY [GLITCH_DOUBLE_RECOVERY ...]]
                         [--glitch-triple-recovery GLITCH_TRIPLE_RECOVERY [GLITCH_TRIPLE_RECOVERY ...]]
                         [--glitch-epoch-range GLITCH_EPOCH_RANGE]
                         [--glitch-td-min GLITCH_TD_MIN]
                         [--glitch-td-max GLITCH_TD_MAX]
                         [--glitch-f0-range GLITCH_F0_RANGE]
                         [--glitch-f1-range GLITCH_F1_RANGE]
                         [--glitch-f2-range GLITCH_F2_RANGE]
                         [--glitch-f0d-range GLITCH_F0D_RANGE]
                         [--glitch-td-range GLITCH_TD_RANGE]
                         [--glitch-f0d-positive]
                         [--glitch-td-split GLITCH_TD_SPLIT [GLITCH_TD_SPLIT ...]]
                         [--glitch-alt-f0]
                         [--glitch-alt-f0t GLITCH_ALT_F0T [GLITCH_ALT_F0T ...]]
                         [--alt-f0t-gltd] [--measured-prior]
                         [--measured-without]
                         [--measured-sigma MEASURED_SIGMA [MEASURED_SIGMA ...]]
                         [--auto-add] [--fit-planets] [--planets PLANETS]
                         [--mass-max MASS_MAX [MASS_MAX ...]]
                         [--mass-min MASS_MIN [MASS_MIN ...]]
                         [--mass-log-prior]
                         [--period-max PERIOD_MAX [PERIOD_MAX ...]]
                         [--period-min PERIOD_MIN [PERIOD_MIN ...]] [--cont]
                         [--nthread NTHREAD] [-N NSAMPLE] [-n] [--nlive NLIVE]
                         [--nwalkers NWALKERS] [--emcee] [--dynesty]
                         [--dynesty-plots]
                         [--dynesty-bound-eff DYNESTY_BOUND_EFF]
                         [--dynesty-bound DYNESTY_BOUND]
                         [--dynesty-sampler DYNESTY_SAMPLER]
                         [--dynesty-bootstrap DYNESTY_BOOTSTRAP] [--zeus]
                         [--multinest] [--multinest-prefix MULTINEST_PREFIX]
                         [--white-corner] [--all-corner] [--plot-chain]
                         [--plot-derived] [--burn BURN]
                         [--truth-file TRUTH_FILE]
                         [--dump-samples DUMP_SAMPLES] [--dump-chain]
                         [--plotname PLOTNAME] [--skip-plots]
                         par tim

Run 'enterprise' on a single pulsar

positional arguments:
  par
  tim

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR, -o OUTDIR
                        Output directory for chains etc

RedNoise:
  RedNoise Parameters. Note that the red noise model is default ENABLED

  --no-red-noise        Disable Power Law Red Noise search
  --Ared-max ARED_MAX, -A ARED_MAX
                        Max log10A_Red
  --Ared-min ARED_MIN   Min log10A_Red
  --red-gamma-max RED_GAMMA_MAX
                        Max gamma red
  --red-gamma-min RED_GAMMA_MIN
                        Min gamma red
  --red-prior-log       Use uniform prior in log space for red noise amplitude
  --red-ncoeff RED_NCOEFF
                        Number of red noise coefficients (nC)
  --tspan-mult TSPAN_MULT
                        Multiplier for tspan
  --qp                  Use QP nudot model
  --qp-ratio-max QP_RATIO_MAX
  --qp-sigma-max QP_SIGMA_MAX
  --qp-p-min-np QP_P_MIN_NP
  --qp-p-min QP_P_MIN
  --qp-p-max QP_P_MAX

WhiteNoise:
  WhiteNoise Parameters. White noise defaults to Enabled.

  --no-white            Disable efac and equad
  --jbo, -j             Use -be flag for splitting backends
  --be-flag BE_FLAG, -f BE_FLAG
                        Use specified flag for splitting backends
  --white-prior-log     Use uniform prior in log space for Equad
  --efac-max EFAC_MAX   Max for efac prior
  --efac-min EFAC_MIN   Min for efac prior
  --ngecorr             Add ECORR for the nanograv backends

DM Variations:
  Parameters for fitting dm variations

  -D, --dm              Enable DM variation search
  --Adm-max ADM_MAX     Max log10A_DM
  --Adm-min ADM_MIN     Min log10A_DM
  --dm-ncoeff DM_NCOEFF
                        Number of DM bins to use
  --dm-prior-log        Use uniform prior in log space for dm noise amplitude
  --dm-tspan-mult DM_TSPAN_MULT
                        Multiplier for tspan for dm
  --dm1 DM1             fit for DM1
  --dm2 DM2             fit for DM2

BasicTimingModel:
  Basic pulsar spin and astrometric parameters.

  --f2 F2               range of f2 to search
  --pm                  Fit for PMRA+PMDEC
  --px                  Fit for parallax
  --px-range PX_RANGE   Max parallax to search
  --pm-angle            Fit for PM + angle rather than by PMRA/PMDEC
  --pm-range PM_RANGE   Search range for proper motion (deg/yr)
  --pm-ecliptic         Generate ecliptic coords for proper motion
  --pos                 Fit for position (linear fit only)
  --pos-range POS_RANGE
                        Search range for position (arcsec)
  --legendre            Fit polynomial using Legendre series
  --leg-df0 LEG_DF0     Max offset in f0 parameters
  --leg-df1 LEG_DF1     Max offset in f1 parameters
  --wrap WRAP [WRAP ...]
                        Fit for missing phase wraps at this epoch
  --wrap-range WRAP_RANGE
                        Max number of wraps missing
  --tm-fit-file TM_FIT_FILE
                        Use setup file for timing model parameters ONLY WORKS
                        WITH Multinest/MPI

GlitchModel:
  Glitch and recovery parameters.

  --glitch-all, --gl-all
                        fit for all glitches
  --glitches GLITCHES [GLITCHES ...]
                        Select glitches to fit
  --glitch-recovery GLITCH_RECOVERY [GLITCH_RECOVERY ...]
                        fit for glitch recoveries on these glitches
  --glitch-double-recovery GLITCH_DOUBLE_RECOVERY [GLITCH_DOUBLE_RECOVERY ...]
                        fit for a second glitch recovery on these glitches
  --glitch-triple-recovery GLITCH_TRIPLE_RECOVERY [GLITCH_TRIPLE_RECOVERY ...]
                        fit for a third glitch recovery on these glitches
  --glitch-epoch-range GLITCH_EPOCH_RANGE, --glep-range GLITCH_EPOCH_RANGE
                        Window for glitch epoch fitting
  --glitch-td-min GLITCH_TD_MIN, --gltd-min GLITCH_TD_MIN
                        Min log(td)
  --glitch-td-max GLITCH_TD_MAX, --gltd-max GLITCH_TD_MAX
                        Max log10(td)
  --glitch-f0-range GLITCH_F0_RANGE, --glf0-range GLITCH_F0_RANGE
                        Fractional change in glF0
  --glitch-f1-range GLITCH_F1_RANGE, --glf1-range GLITCH_F1_RANGE
                        Fractional change in glF1
  --glitch-f2-range GLITCH_F2_RANGE, --glf2-range GLITCH_F2_RANGE
                        Fractional change in glF2
  --glitch-f0d-range GLITCH_F0D_RANGE, --glf0d-range GLITCH_F0D_RANGE
                        Fractional range of f0d compared to glF0
  --glitch-td-range GLITCH_TD_RANGE, --gltd-range GLITCH_TD_RANGE
                        Fractional change in gltd
  --glitch-f0d-positive, --glf0d-positive
                        Allow only positive GLF0D
  --glitch-td-split GLITCH_TD_SPLIT [GLITCH_TD_SPLIT ...]
                        Where to split the td prior for multi exponentials
  --glitch-alt-f0       Use alternative parameterisation of glitches fitting
                        for instantanious change in F0 rather than GLF0
  --glitch-alt-f0t GLITCH_ALT_F0T [GLITCH_ALT_F0T ...]
                        Replace GLF1 with change of spin frequency 'T' days
                        after the glitches respectively
  --alt-f0t-gltd        Replace GLF1 with change of spin frequency 'gltd' days
                        after the glitch for all glitches
  --measured-prior      Use measured prior range for GLF0(instant) and
                        GLF0(T=taug)
  --measured-without    Measure prior range without glitch for GLF0(instant)
                        and GLF0(T=taug)
  --measured-sigma MEASURED_SIGMA [MEASURED_SIGMA ...], --sigma-range MEASURED_SIGMA [MEASURED_SIGMA ...]
                        Minus/Plus sigma range of GLF0(instant), and
                        Minus/Plus sigma range of GLF0(T=taug) respectively
  --auto-add            Automatic add all existing glitches recoveries in the
                        par file

Planet:
  Planet Orbital parameters.

  --fit-planets, -P     Fit for 1st planet orbit parameters
  --planets PLANETS     Number of planets to fit
  --mass-max MASS_MAX [MASS_MAX ...]
                        Max mass (Earth masses) prior for planets. Each planet
                        fitted needs a value for this.
  --mass-min MASS_MIN [MASS_MIN ...]
                        Min mass (Earth masses) prior for planets. Each planet
                        fitted needs a value for this.
  --mass-log-prior      Use log mass prior for all planets.
  --period-max PERIOD_MAX [PERIOD_MAX ...]
                        Max period (days) prior for planets. Each planet
                        fitted needs a value for this.
  --period-min PERIOD_MIN [PERIOD_MIN ...]
                        Min period (days) prior for planets. Each planet
                        fitted needs a value for this.

Sampling options:
  --cont                Continue existing run
  --nthread NTHREAD, -t NTHREAD
                        number of threads
  -N NSAMPLE, --nsample NSAMPLE
                        (max) number of samples
  -n, --no-sample       Disable the actual sampling...
  --nlive NLIVE         Number of live points (nested)
  --nwalkers NWALKERS   number of walkers (mcmc)

EmceeSolver:
  Configure for mcmc with EMCEE

  --emcee               Use emcee sampler

DynestySolver:
  Configure for nested sampling with DyNesty

  --dynesty             Use dynesty sampler
  --dynesty-plots       make dynesty run plots
  --dynesty-bound-eff DYNESTY_BOUND_EFF
                        Efficiency to start bounding
  --dynesty-bound DYNESTY_BOUND
                        Bounding method
  --dynesty-sampler DYNESTY_SAMPLER
                        Sampling method
  --dynesty-bootstrap DYNESTY_BOOTSTRAP
                        Bootstrap amount

ZeusSolver:
  Configure for mcmc with Zeus

  --zeus                Use zeus sampler

MultiNestSolver:
  Configure for nested sampling with pymultinest

  --multinest           Use pymultinest sampler
  --multinest-prefix MULTINEST_PREFIX
                        Prefix for pymultinest runs

Output options:
  --white-corner        Make the efac/equad corner plots
  --all-corner, --corner-all
                        Make corner plots with all params
  --plot-chain          Make a plot of the chains/posterior samples
  --plot-derived        Include derived parameters in corner plots etc.
  --burn BURN           Fraction of chain to burn-in (MC only; default=0.25)
  --truth-file TRUTH_FILE
                        Truths values of parameters; maxlike=maximum
                        likelihood, default=None
  --dump-samples DUMP_SAMPLES
                        Dump N samples of final scaled parameters
  --dump-chain          Dump chain after scaling to phyical parameters
  --plotname PLOTNAME   Set plot output file name stem
  --skip-plots          Skip all plotting (for debugging I guess)

```