#!/usr/bin/env python

"""
Code to launch temponest jobs for Fermi MSPs to do noise modelling and model selection
a) GWB with RN, WN
b) GWB with RN
c) GWB with WN
d) GWB only
"""

#IMPORTS
import argparse
import numpy as np
import sys
import glob
import os
import subprocess
import shlex
import time


#TempoNest: Fit for RAJ,DECJ,F0 and F1 as custom priors
def temponest(pulsar,par,tim,cfile):
    """
    pulsar: Full path to the pulsar directory
    par: Path to step1.par
    tim: Path to step1.tim
    cfile: Path to basicparams.cfile
    """

    #Identifying the directory name from the input config file
    config_file = open(cfile,"r")
    cfile_lines = config_file.readlines()
    config_file.close()
    for line in cfile_lines:
        sline = line.split()
        if len(sline) > 0:
            if sline[0] == "root":
                dirname = sline[2].split("/")[0]

    #Creating the TN results directory
    psrpath,psrname = os.path.split(pulsar)
    if not os.path.exists(os.path.join(pulsar,dirname)):
        print ("Creating the TN results : {0}".format(dirname))
        os.makedirs(os.path.join(pulsar,dirname))
    tndir = os.path.join(pulsar,dirname)

    #Creating the bash script for the job
    job_name = str(psrname)+"_"+"{0}.bash".format(dirname)
    if not os.path.exists(os.path.join(tndir,job_name)):
        with open(os.path.join(tndir,job_name),'w') as job_file:
           
            job_file.write("#!/bin/bash \n")
            job_file.write("#SBATCH --job-name={0}\n".format(job_name))
            job_file.write("#SBATCH --output={0}/{1}.out \n".format(tndir,job_name))
            job_file.write("#SBATCH --ntasks=28 \n")
            job_file.write("#SBATCH --time=07:00:00 \n")
            job_file.write("#SBATCH --mem-per-cpu=4g \n")
            job_file.write("#SBATCH --mail-type=FAIL --mail-user=adityapartha3112@gmail.com \n")
            job_file.write('cd {0} \n'.format(pulsar))
            job_file.write("tempo2 -gr temponest -f {0} {1} -cfile {2}".format(par,tim,cfile))
            job_file.close()

        print "Job bash file created"

    else:
        print "Job exists for {0}:{1}".format(psrname,job_name)

    #Deploying the job
    if not os.path.exists(os.path.join(tndir,"{0}.out".format(job_name))):
        com_sbatch = 'sbatch {0}/{1}'.format(tndir,job_name)
        args_sbatch = shlex.split(com_sbatch)
        proc_sbatch = subprocess.Popen(args_sbatch)
        print ("{0} deployed".format(job_name))


def temponest_cadence(pulsar,par,tim,cfile,cadence):
    """
    pulsar: Full path to the pulsar directory
    par: Path to step1.par
    tim: Path to step1.tim
    cfile: Path to basicparams.cfile
    Optional: cadence: The temponest results are prefixed with the specified cadence
    """

    #Identifying the directory name from the input config file
    config_file = open(cfile,"r")
    cfile_lines = config_file.readlines()
    config_file.close()
    for line in cfile_lines:
        sline = line.split()
        if len(sline) > 0:
            if sline[0] == "root":
                dirname = sline[2].split("/")[0]

   
    #Creating the TN results directory
    psrpath,psrname = os.path.split(pulsar)
    cadence_path = os.path.join(pulsar,cadence)
    if not os.path.exists(os.path.join(cadence_path,dirname)):
        print ("Creating the TN results inside the specified cadence directory : {0}:{1}".format(cadence,dirname))
        os.makedirs(os.path.join(cadence_path,dirname))
    tndir = os.path.join(cadence_path,dirname)

    #Creating the bash script for the job
    job_name = str(psrname)+"_"+"{0}_{1}.bash".format(cadence,dirname)
    if not os.path.exists(os.path.join(tndir,job_name)):
        with open(os.path.join(tndir,job_name),'w') as job_file:
           
            job_file.write("#!/bin/bash \n")
            job_file.write("#SBATCH --job-name={0}\n".format(job_name))
            job_file.write("#SBATCH --output={0}/{1}.out \n".format(tndir,job_name))
            job_file.write("#SBATCH --ntasks=28 \n")
            job_file.write("#SBATCH --time=07:00:00 \n")
            job_file.write("#SBATCH --mem-per-cpu=4g \n")
            job_file.write("#SBATCH --mail-type=FAIL --mail-user=adityapartha3112@gmail.com \n")
            job_file.write('cd {0} \n'.format(cadence_path))
            job_file.write("srun --ntasks=28 --mem-per-cpu=4g tempo2 -gr temponest -f {0} {1} -cfile {2}".format(par,tim,cfile))
            job_file.close()

        print "Job bash file created"

    else:
        print "Job exists for {0}:{1}".format(psrname,job_name)

    #Deploying the job
    if not os.path.exists(os.path.join(tndir,"{0}.out".format(job_name))):
        com_sbatch = 'sbatch {0}/{1}'.format(tndir,job_name)
        args_sbatch = shlex.split(com_sbatch)
        proc_sbatch = subprocess.Popen(args_sbatch)
        print ("{0} deployed".format(job_name))



parser=argparse.ArgumentParser(description="TempoNest pulsar timer")
parser.add_argument("-data",dest="data_path",help="Path to the data directory (should have $psr/*TNest.par,*toa")
parser.add_argument("-par", dest="par", help="Extension of the parfile to use")
parser.add_argument("-tim", dest="tim", help="Extension of the tim file to use")
parser.add_argument("-cfile", dest="cfile", help="Path to config file directory")
parser.add_argument("-cadence", dest="cadence", help="Explicitly specify ToA cadence to distinguish Temponest output files")
parser.add_argument("-pulsar", dest="pulsar", help="Run analysis only for the specified pulsar")
args=parser.parse_args()


pulsars = sorted(glob.glob(os.path.join(args.data_path,"J*")))

for pulsar in pulsars:

    if args.pulsar:

        if str(args.pulsar) in pulsar:

            par = glob.glob(os.path.join(pulsar,"*{0}*".format(args.par)))[0]
            tim = glob.glob(os.path.join(pulsar,"*{0}*".format(args.tim)))[0]

            psrname = os.path.split(pulsar)[-1]
            print "================================== {0} ==================================".format(psrname)
            print par
            print tim

            cfiles = sorted(glob.glob(os.path.join(args.cfile,"*cfile")))

            print "===================================================="
            for cfile in cfiles:
                print cfile
                if args.cadence:
                    cadence = str(args.cadence)
                    temponest_cadence(pulsar,par,tim,cfile,cadence)
                else:
                    temponest(pulsar,par,tim,cfile)
                
                print "------------------------"

            print "============================== END ==========================================="

    else:

        par = glob.glob(os.path.join(pulsar,"*{0}*".format(args.par)))[0]
        tim = glob.glob(os.path.join(pulsar,"*{0}*".format(args.tim)))[0]

        psrname = os.path.split(pulsar)[-1]
        print "================================== {0} ==================================".format(psrname)
        print par
        print tim

        cfiles = sorted(glob.glob(os.path.join(args.cfile,"*cfile")))

        #"""
        print "===================================================="
        for cfile in cfiles:
            print cfile
            if args.cadence:
                cadence = str(args.cadence)
                temponest_cadence(pulsar,par,tim,cfile,cadence)
            else:
                temponest(pulsar,par,tim,cfile)
            
            print "------------------------"

        print "============================== END ==========================================="
        #"""




