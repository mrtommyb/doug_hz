#! /usr/bin/env python

import os
import sys
import numpy as np 

#import mcmc_rho_run_hdf5_doug_pl
import subprocess

def main(star,local=False,dil=None):
    if not local:
        codedir='/nobackupp1/tsbarcl2/Projects/doug_hz/code'
        sys.path.append(codedir)
        maindir = '/nobackupp1/tsbarcl2/Projects/doug_hz/data'
        import mcmc_rho_run_hdf5_doug_pl
    elif local:
        codedir = '/Users/tom/Projects/doug_hz/code'
        sys.path.append(codedir)
        maindir = '/Users/tom/Projects/doug_hz/data'
        import mcmc_rho_run_hdf5_doug_pl


    os.chdir(maindir + '/' + star + '/')

    files = os.listdir('.')
    nfile = filter(lambda x: x.endswith('n1.dat'), files)[0]

    wc = subprocess.check_output(['wc','-l',nfile])
    lines = int(wc.split()[0])
    if lines == 18:
        nplanets = 1
    elif lines == 28:
        nplanets = 2
    elif lines == 38:
        nplanets = 3
    elif lines == 48:
        nplanets = 4
    elif lines == 58:
        nplanets = 5
    elif lines == 68:
        nplanets = 6

    #run 1200 * 2000 * nplanets mcmc chains

    sampler = mcmc_rho_run_hdf5_doug_pl.main(
        nw=200,th=6,bi=5,fr=5*nplanets,thin=2,
        local=local,runmpi=False,dil=dil,
        codedir=codedir,
        ldfileloc=codedir + '/')

    outfile = open(maindir + '/done.dat', 'a')
    outfile.write(str(star) + ' \n')
    outfile.close()

if __name__ == '__main__':
    star = sys.argv[1]
    if len(sys.argv) >2 and sys.argv[2] == 'local':
        local = True
        if len(sys.argv) >3:
            dil = float(sys.argv[3])
    elif len(sys.argv) >2:
            dil = float(sys.argv[2])
            local = False
    else:
        dil=None
        local = False
        
    main(star,local=local,dil=dil)


