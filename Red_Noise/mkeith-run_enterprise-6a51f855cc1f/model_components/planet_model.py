from . import xparameter as parameter
import numpy as np
#from enterprise.signals import utils
#from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import signal_base
import os

#import fit_planet
import scipy.interpolate as interpolate
from astropy import coordinates as coord
from astropy import units as units
from astropy import constants as aconst

name="Planet"
argdec="Planet Orbital parameters."

from inspect import getsourcefile
from os.path import abspath

path_to_here = abspath(getsourcefile(lambda:0))
path_to_ecc = path_to_here.split('planet_model.py')[0]



def setup_argparse(parser):

   parser.add_argument('--fit-planets','-P',action='store_true', help='Fit for 1st planet orbit parameters')
   parser.add_argument('--planets', type=int,default=1, help='Number of planets to fit')

   parser.add_argument('--mass-max',type=float,default=[], nargs='+', help='Max mass (Earth masses) prior for planets. Each planet fitted needs a value for this.')
   parser.add_argument('--mass-min',type=float,default=[], nargs='+',help='Min mass (Earth masses) prior for planets. Each planet fitted needs a value for this.')
   parser.add_argument('--mass-log-prior',action='store_true',help='Use log mass prior for all planets.')
   parser.add_argument('--period-max',type=float,default=[], nargs='+',help='Max period (days) prior for planets. Each planet fitted needs a value for this.')
   parser.add_argument('--period-min',type=float,default=[], nargs='+', help='Min period (days) prior for planets. Each planet fitted needs a value for this.')
   parser.add_argument('--ecc-log-prior',action='store_true',help='Use log ecc prior for all planets.')

#   parser.add_argument('--planet2','-P2',action='store_true', help='Fit for 2nd planet orbit parameters')
#   parser.add_argument('--mass-max2',type=float,default=1,help='Max mass (Earth masses) 2nd planet')
#   parser.add_argument('--mass-min2',type=float,default=1e-3,help='Min mass (Earth masses) 2nd planet')
#   parser.add_argument('--mass-log-prior2',action='store_true',help='Use log prior for 2nd planet mass')
#   parser.add_argument('--period-max2',type=float,default=3000,help='Max period (days) 2nd planet')
#   parser.add_argument('--period-min2',type=float,default=50,help='Min period (days) 2nd planet')


def setup_model(args, psr, parfile):

   components = []

#   print('!!!!!!!!!!!!',args.planets,args.mass_max)   

   if args.fit_planets:
      n = args.planets #number of planets to fit
   
      for i in range(n):
        
         if args.mass_log_prior:
            if args.ecc_log_prior:
               planet = fit_planet_app2(psrMass=parameter.Constant(1.4), mass=parameter.Uniform(np.log10(args.mass_min[i]),np.log10(args.mass_max[i])), period=parameter.Uniform(args.period_min[i],args.period_max[i]), phase=parameter.Uniform(0,1), omega=parameter.Uniform(0,360), ecc=parameter.Uniform(-9,np.log10(0.9)),ismasslog=True,isecclog = True)
            else:
               planet = fit_planet_app2(psrMass=parameter.Constant(1.4), mass=parameter.Uniform(np.log10(args.mass_min[i]),np.log10(args.mass_max[i])), period=parameter.Uniform(args.period_min[i],args.period_max[i]), phase=parameter.Uniform(0,1), omega=parameter.Uniform(0,360), ecc=parameter.Uniform(0,0.9),ismasslog = True)

         else:
            if args.ecc_log_prior:
               planet = fit_planet_app2(psrMass=parameter.Constant(1.4), mass=parameter.Uniform(args.mass_min[i],args.mass_max[i]), period=parameter.Uniform(args.period_min[i],args.period_max[i]), phase=parameter.Uniform(0,1), omega=parameter.Uniform(0,360), ecc=parameter.Uniform(-9,np.log10(0.9)),isecclog=True)
            else:
               planet = fit_planet_app2(psrMass=parameter.Constant(1.4), mass=parameter.Uniform(args.mass_min[i],args.mass_max[i]), period=parameter.Uniform(args.period_min[i],args.period_max[i]), phase=parameter.Uniform(0,1), omega=parameter.Uniform(0,360), ecc=parameter.Uniform(0,0.9))

         planet_signal = deterministic_signals.Deterministic(planet,name='planet'+str(i+1))

         components.append(planet_signal)


   if len(components) > 0:
      model = components[0]
      for m in components[1:]:
         model += m
      return model
   else:
      return None


def to_par(self,p,chain):
   if "planet_ecc" in p:
      return "planet_ecc", chain
   if "planet_mass" in p:
      return "planet_mass", chain
   if "planet_period" in p:
      return "planet_period", chain
   if "planet_phase" in p:
      return "planet_phase", chain
   if "planet_omega" in p:
      return "planet_omega", chain

   else:
      return None




@signal_base.function
def fit_planet_app2(toas,psrMass,mass,period,phase,omega,ecc,ismasslog = False,isecclog=False):
    # replace the highly-degenerate time of periastron
    # with a 'phase', i.e true anomaly/2pi, between [0,1), at a specific time,
    # e.g. 55000 MJD
    if ismasslog:
        mass = 10**mass
    if isecclog:
        ecc = 10**ecc

    Omega_b = 2.0*np.pi/(units.day*period)
    e = ecc

    tref = 55000 #MJD
    tref = tref*units.day
    if phase <= 0.5:
        Eref = np.arccos((e+np.cos(2*np.pi*phase))/(1+e*np.cos(2*np.pi*phase)))
    else:
        Eref = 2*np.pi-np.arccos((e+np.cos(2*np.pi*phase))/(1+e*np.cos(2*np.pi*phase)))

    mean_anom_ref = Eref - e*np.sin(Eref)
    t0 = tref - mean_anom_ref/Omega_b #should be in sec

    #inc=inc*units.degree

    M1 = psrMass * units.M_sun
    M2 = mass * units.M_earth
    Mtot = M1+M2
    Mr = M2**3 / Mtot**2
    a1 = np.power(Mr*aconst.G/Omega_b**2,1.0/3.0).to(units.m)
    #asini = a1 * np.sin(inc)
    asini = a1
    om=coord.Angle(omega*units.deg)

    def get_roemer(t):

        def ecc_anom(E,e,M):
            return (E-e*np.sin(E))-M

        mean_anom = coord.Angle((Omega_b * (t*units.s - t0)).decompose().value*units.rad)

        mean_anom.wrap_at(2*np.pi*units.rad,inplace=True)
        mean_anom = mean_anom.rad
        
        ### read/interp E solution from appropriate ecc file:
        
        e_app = np.around(e,decimals=5)
        #print('\n????? Dir: ',os.getcwd())

        sampled_mean_anom,sampled_E = np.load(path_to_ecc+'E_e_5dig/E_e=%.5f.npy' %e_app)
        E_from_m = interpolate.interp1d(sampled_mean_anom,sampled_E,copy=False,kind='cubic')
        #if (mean_anom<1e-6).any():
        #    print(e,e_app)
        #    print(mean_anom[mean_anom<1e-6])
	#print('-------- MEAN ANOM: '+str(mean_anom))
        #print('-------- min Sampled: '+str(min(sampled_mean_anom)))
        E = E_from_m(mean_anom)
 
        roemer = (asini*(np.cos(E)-e)*np.sin(om) + asini*np.sin(E)*np.sqrt(1.0-e**2)*np.cos(om))/aconst.c

        return roemer

    roemer = get_roemer(toas)
    return roemer.to(units.s).value


