import numpy as np
import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os, sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


print ('CAUTION: The model is capable of accounting for only one exponential recovery after the glitch, if any. For mutiple exponential recoveries the code will give wrong results!' )

### make all the functions used for the computation 
def glitch_phase(t,phi, nu_p, nu_p_dot, tau_d, nu_d, t_g):
    dt = (t-t_g)*86400.0
    tau_d = tau_d*86400.0
    if tau_d !=0:
        phase = phi + nu_p*dt +0.5*nu_p_dot*dt**2 +(1.0-np.exp(-dt/tau_d))*nu_d*tau_d
    else:
        phase = phi + nu_p*dt +0.5*nu_p_dot*dt**2
    return phase

def compute_epoch(phi, nu_p, nu_p_dot, tau_d, nu_d, t_g, t_g_err, t_g_start, t_g_end, phi_err, nu_p_err, nu_p_dot_err, tau_d_err, nu_d_err):
    dist_tg = []
    nu_p_sim = np.random.normal(nu_p,nu_p_err/2.,N_samp)
    nu_p_dot_sim = np.random.normal(nu_p_dot, nu_p_dot_err/2.,N_samp)
    tau_d_sim = np.random.normal(tau_d, tau_d_err/2.,N_samp)
    nu_d_sim = np.random.normal(nu_d,nu_d_err/2.,N_samp)
    phi_sim = np.random.normal(phi, phi_err/2.,N_samp)
    All_param_sim = np.array([nu_p_sim, nu_p_dot_sim, tau_d_sim, nu_d_sim, phi_sim])
    for k in range(N_samp):
        root = fsolve(glitch_phase, [t_g_start, t_g_end], args=(phi_sim[k], nu_p_sim[k], nu_p_dot_sim[k], tau_d_sim[k], nu_d_sim[k], t_g))
        dist_tg.append(root[0])
    return dist_tg, All_param_sim




def find_epoch_range(phi, nu_p, nu_p_dot, tau_d, nu_d, GLEP,results, t_start, t_stop, plot_option, psr, All_param_sim):
    t_plot = np.linspace(t_start-50, t_stop+50,300)
    t = np.linspace(t_start, t_stop, 300)
    solutions =[]
    all_vals =[]
    for i in range(len(results)):
        nu_p_sim = All_param_sim[0][i]
        nu_p_dot_sim = All_param_sim[1][i]
        tau_d_sim = All_param_sim[2][i]
        nu_d_sim = All_param_sim[3][i]
        phi_sim = All_param_sim[4][i]
        
        t_g = results[i]
        phase = glitch_phase(t,0.0,nu_p_sim,nu_p_dot_sim,tau_d_sim,nu_d_sim, t_g)
        phase_plot = glitch_phase(t_plot,0.0,nu_p_sim,nu_p_dot_sim,tau_d_sim,nu_d_sim, t_g)
        int_part_phase = phase.astype(int).astype(float) ## finding the integer part of the phase
        min_phase = np.min(int_part_phase)
            
        max_phase = np.max(int_part_phase)
        if max_phase != 0.0 or min_phase!=0.0:
            int_phases_array = np.arange(min_phase,max_phase+1)
        else:
            int_phases_array =np.array([0.0])
        
        for k in range(len(int_phases_array)):
            roots = fsolve(glitch_phase, t_g, args=(0-int_phases_array[k], nu_p_sim, nu_p_dot_sim, tau_d_sim, nu_d_sim, t_g))[0]
            all_vals.append(roots)
        if plot_option =='Y':
            flname = psr+'.'+str(GLEP)+'.eph.error.pdf'
            plt.plot(t_plot,phase_plot,color='grey', alpha=0.3)
            
    if np.mean(int_phases_array) !=0.0:
        all_vals = np.array(all_vals)
        locs_log = np.logical_and(all_vals>=t_start, all_vals<=t_stop)
        locs = np.where(locs_log==True)[0]
        solutions_sel = all_vals[locs]
        max_sol_samp = np.max(solutions_sel)
        min_sol_samp = np.min(solutions_sel)
            
    if np.mean(int_phases_array)==0.0:
        solutions_sel = all_vals
            
            
    if plot_option=='Y':
        plt.title(psr+' GLEP '+str(GLEP))
        phase_m = glitch_phase(t_plot,phi,nu_p,nu_p_dot,tau_d,nu_d, GLEP)
        plt.plot(t_plot, phase_m, color='red', label='Timing Model')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('MJD')
        plt.ylabel(r'Glitch phase model $(\phi_g)$')
        plt.axvline(t_start, linestyle='--', color='blue', label='Bounds')
        plt.axvline(t_stop, linestyle='--', color='blue')
        plt.axvline(GLEP, label='Initial tg')
        if np.mean(int_phases_array) !=0.0:
            max_sol = np.max(max_sol_samp)
            min_sol = np.min(min_sol_samp)
            plt.axvline(max_sol, linestyle='-.', color='orange', label='solution')
            plt.axvline(min_sol, linestyle='-.', color='orange')
            central_value = 0.5*(max_sol+min_sol)
            edge = 0.5*(max_sol-min_sol)
            
        if np.mean(int_phases_array)==0.0:
            plt.axvline(np.mean(solutions_sel), color='k')
            plt.axvline(np.mean(solutions_sel)+np.std(solutions_sel), color='k', linestyle='--')
            plt.axvline(np.mean(solutions_sel)-np.std(solutions_sel), color='k', linestyle='--')
            min_value = np.min(solutions_sel)
            max_value = np.max(solutions_sel)
            central_value = 0.5*(min_value + max_value)
            edge = 0.5*(max_value-min_value)
            
            
        plt.grid()
        plt.legend()
    
    plt.savefig(flname)
    plt.clf()
        
    print ('Glitch epoch:', central_value,'Uncertainity:', edge)
    return flname, central_value, edge

N_samp=500



glitch_param_data = np.loadtxt('Input.data.sample.csv', delimiter=',', dtype=str)
psr = glitch_param_data[:,0]
GLEP = glitch_param_data[:,1].astype(float)
df_by_f = glitch_param_data[:,3].astype(float)
df_by_f_err = glitch_param_data[:,4].astype(float)
df1_by_f1 = glitch_param_data[:,5].astype(float)
df1_by_f1_err = glitch_param_data[:,6].astype(float)
df0 = glitch_param_data[:,9].astype(float)
df0_err = glitch_param_data[:,10].astype(float)
df1= glitch_param_data[:,13].astype(float)
df1_err= glitch_param_data[:,14].astype(float)
dfd = glitch_param_data[:,11].astype(float)
dfd_err = glitch_param_data[:,12].astype(float)
tau = glitch_param_data[:,15].astype(float)
tau_err =glitch_param_data[:,16].astype(float)
TOA_low = glitch_param_data[:,17].astype(float)
TOA_high = glitch_param_data[:,18].astype(float)
GLPH = glitch_param_data[:,20].astype(float)
GLPH_err = glitch_param_data[:,21].astype(float)


fls =[]
pulsar =[]
epoch =[]
epoch_err =[]
jumps =[]
jumps_err =[]
jumps_dot =[]
jumps_dot_err =[]
for k in range(len(psr)):
    print('Analysis glitch number:', k+1)
    #if GLPH_err[k] ==0.0:
        #print('GLPH err: 0.0 & PSR is J',psr[k], 'GLEP=', GLEP[k])
    results, all_param_sim = compute_epoch(GLPH[k],df0[k],df1[k],tau[k],dfd[k],GLEP[k],0.0,TOA_low[k],TOA_high[k],GLPH_err[k],df0_err[k],df1_err[k],tau_err[k],dfd_err[k])
    file, cen, ed = find_epoch_range(GLPH[k],df0[k],df1[k],tau[k],dfd[k],GLEP[k],results,TOA_high[k],TOA_low[k],'Y', psr[k], all_param_sim)
    pulsar.append(psr[k])
    epoch.append(cen)
    epoch_err.append(ed)
    jumps.append(df_by_f[k])
    jumps_err.append(df_by_f_err[k])
    jumps_dot.append(df1_by_f1[k])
    jumps_dot_err.append(df1_by_f1_err[k])
    fls.append(file)
str_l = 'pdfunite'
for j in fls:
    str_l = str_l+' '+j
os.system(str_l+' merged.eph.err.pdf')
os.system('rm *.eph.error.pdf')


dats = list(zip(epoch, epoch_err, jumps, jumps_err, jumps_dot, jumps_dot_err))
np.savetxt('tvals.txt', dats, fmt=" %5.8f %5.5f %5.10f %5.10f %5.10f %5.10f", delimiter='\t')

np.savetxt('psrs.txt', list(pulsar), fmt='%s')

np.savetxt('header.txt', ['#col1 = PSR', '#col2 = GLEP', '#col3 = GLEP_err', '#col4 = df0/f0(e-9)', '#col5 = df0/f0_err(e-9)', '#col6 = df1/f1(e-3)', '#col7 = df1/f1_err(e-9)'], fmt ='%s', delimiter='\t')

os.system('paste psrs.txt tvals.txt > int.txt ')
os.system('cat header.txt int.txt > Final.table.txt')
os.system('rm header.txt int.txt psrs.txt tvals.txt')


