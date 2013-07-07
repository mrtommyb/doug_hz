
import numpy as np 
import sys

import transitemcee as tmod 
import emcee
import time as thetime
import os
#import matplotlib.pyplot as plt
from emcee.utils import MPIPool
from scipy.stats import scoreatpercentile as scpc
import h5py
from scipy.stats import nanmedian, nanstd

def get_data(files,cadence=1625.3):
    dat_dict = {}
    #dat_dict['koinum'] = os.path.realpath('.').split('koi')[1]

    dat_dict['cadence'] = cadence

    #lightcurve = filter(lambda x: x.startswith('klc'), files)
    lightcurve = [
        x for x in files if x.startswith('klc') and 'pdc' in x]
    nfile = filter(lambda x: x.endswith('n1.dat'), files)
    stardata = filter(lambda x: x.startswith('stellar'), files)

    lcdat = np.genfromtxt(lightcurve[0]).T 
    dat_dict['time'] = lcdat[0].astype('float')
    dat_dict['flux'] = lcdat[1].astype('float')
    dat_dict['err'] = lcdat[2].astype('float')

    stardat = np.genfromtxt(stardata[0])
    dat_dict['koi'] = stardat[0].astype('int')
    dat_dict['teff'] = stardat[1].astype('float')
    dat_dict['teff_unc'] = stardat[2].astype('float')
    dat_dict['feh'] = stardat[3].astype('float')
    dat_dict['feh_unc'] = stardat[4].astype('float')
    dat_dict['rho'] = stardat[5].astype('float')
    dat_dict['rho_unc'] = stardat[6].astype('float')
    dat_dict['rad'] = stardat[7].astype('float')
    dat_dict['rad_unc'] = stardat[8].astype('float')
    dat_dict['mass'] = stardat[9].astype('float')
    dat_dict['mass_unc'] = stardat[10].astype('float')

    G = 6.67E-11
    msun = 1.98892E30
    rsun = 695500000.
    #get logg with uncertainties
    mass_rand = np.random.normal(dat_dict['mass'],dat_dict['mass_unc'],size=50000)
    rad_rand = np.random.normal(dat_dict['rad'],dat_dict['rad_unc'],size=50000)
    #make mass and rad positive
    mask_pos = np.logical_and(mass_rand > 0.0,rad_rand > 0.0)
    mass_rand = mass_rand[mask_pos]
    rad_rand = rad_rand[mask_pos]
    
    logg_rand = np.log10((G*mass_rand*msun) / (rad_rand * rsun)**2 * 100)

    dat_dict['logg'] = nanmedian(logg_rand) #the median
    dat_dict['logg_unc'] = nanstd(logg_rand) #the uncertainty

    ndat = np.genfromtxt(nfile[0],usecols=(1))
    if len(ndat) == 18:
        dat_dict['nplanets'] = 1
    elif len(ndat) == 28:
        dat_dict['nplanets'] = 2
    elif len(ndat) == 38:
        dat_dict['nplanets'] = 3
    elif len(ndat) == 48:
        dat_dict['nplanets'] = 4
    elif len(ndat) == 58:
        dat_dict['nplanets'] = 5
    elif len(ndat) == 68:
        dat_dict['nplanets'] = 6

    dat_dict['T0_guess'] = ndat[np.arange(dat_dict['nplanets']) * 10 + 8]
    dat_dict['per_guess'] = ndat[np.arange(dat_dict['nplanets']) * 10 + 9]
    dat_dict['b_guess'] = ndat[np.arange(dat_dict['nplanets']) * 10 + 10]
    dat_dict['rprs_guess'] = ndat[np.arange(dat_dict['nplanets']) * 10 + 11]
    dat_dict['ecosw_guess'] = np.zeros(dat_dict['nplanets'])
    dat_dict['esinw_guess'] = np.zeros(dat_dict['nplanets'])

    dat_dict['sol_guess'] = np.zeros(dat_dict['nplanets']*6)
    for i in np.arange(dat_dict['nplanets']):
        dat_dict['sol_guess'][i*6] = dat_dict['T0_guess'][i]
        dat_dict['sol_guess'][i*6 + 1] = dat_dict['per_guess'][i]
        dat_dict['sol_guess'][i*6 + 2] = dat_dict['b_guess'][i]
        dat_dict['sol_guess'][i*6 + 3] = dat_dict['rprs_guess'][i]
        dat_dict['sol_guess'][i*6 + 4] = dat_dict['ecosw_guess'][i]
        dat_dict['sol_guess'][i*6 + 5] = dat_dict['esinw_guess'][i]


    return dat_dict 

def main(nw=1000,th=9,bi=500,fr=2000,thin=20,runmpi=True,local=False,
    dil=None,codedir='/Users/tom/Projects/doug_hz/code',
         ldfileloc='/Users/tom/Projects/doug_hz/code/'):
    if runmpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool=None

    #if not local:
        #sys.path.append('/u/tsbarcl2/svn_code/tom_code/')
        #ldfileloc = '/u/tsbarcl2/svn_code/tom_code/'
    #elif local:
        #sys.path.append('/Users/tom/svn_code/tom_code/')
        #ldfileloc = '/Users/tom/svn_code/tom_code/'


    if dil is None:
        dil = 0.0

    files = os.listdir('.')
    dat_d = get_data(files)

    rho_prior = True
    ldp_prior = False

    #mcmc params
    nwalkers = nw
    threads = th
    burnin = bi
    fullrun = fr

    #use quadratic or 4 parameter limb darkening
    n_ldparams = 2

    #lc time offset from BJD-24548333. 
    toffset = (54832.5 + 67.)

    #photometric zeropoint
    zpt_0 = 1.E-10

    #plot?
    #doplot=False

    ################

    M = tmod.transitemcee_fitldp(dat_d['nplanets'],dat_d['cadence'],
        ldfileloc=ldfileloc)

    #M.get_stellar(dat_d['teff'],dat_d['logg'],dat_d['feh'],n_ldparams)

    M.get_stellar(dat_d['teff'],
        dat_d['logg'],
        dat_d['feh'],
        n_ldparams,ldp_prior=ldp_prior)

    M.already_open(dat_d['time'],
        dat_d['flux'],dat_d['err'],
        timeoffset=toffset,normalize=False)

    rho_vals = np.array([dat_d['rho'],dat_d['rho_unc']])
    M.get_rho(rho_vals,rho_prior)
    M.get_zpt(zpt_0)

    if dil is not None:
        M.get_sol(*dat_d['sol_guess'],dil=dil)
    else:
        M.get_sol(*dat_d['sol_guess'])

    M.cut_non_transit(8)

    ################
    stophere = False
    if not stophere:

    #for threadnum in np.arange(2,32,2):
        p0 = M.get_guess(nwalkers)
        l_var = np.shape(p0)[1]

        N = len([indval for indval in xrange(fullrun)
                if indval%thin == 0])
        outfile = 'koi{0}_np{1}_prior{2}_dil{3}.hdf5'.format(
            dat_d['koi'],dat_d['nplanets'],rho_prior,dil)
        with h5py.File(outfile, u"w") as f:
            f.create_dataset("time", data=M.time)
            f.create_dataset("flux", data=M.flux)
            f.create_dataset("err", data=M.err)
            f.create_dataset("itime", data=M._itime)
            f.create_dataset("ntt", data = M._ntt)
            f.create_dataset("tobs", data = M._tobs)
            f.create_dataset("omc",data = M._omc)
            f.create_dataset("datatype",data = M._datatype)
            f.attrs["rho_0"] = M.rho_0
            f.attrs["rho_0_unc"] = M.rho_0_unc
            f.attrs["nplanets"] = M.nplanets
            f.attrs["ld1"] = M.ld1
            f.attrs["ld2"] = M.ld2
            f.attrs["koi"] = dat_d['koi']
            f.attrs["dil"] = dil
            g = f.create_group("mcmc")
            g.attrs["nwalkers"] = nwalkers
            g.attrs["burnin"] = burnin
            g.attrs["iterations"] = fullrun
            g.attrs["thin"] = thin
            g.attrs["rho_prior"] = rho_prior
            g.attrs["ldp_prior"] = ldp_prior
            g.attrs["onlytransits"] = M.onlytransits
            g.attrs["tregion"] = M.tregion
            g.attrs["ldfileloc"] = M.ldfileloc
            g.attrs["n_ldparams"] = M.n_ldparams
            g.create_dataset("fixed_sol", data= M.fixed_sol)
            g.create_dataset("fit_sol_0", data= M.fit_sol_0)


            c_ds = g.create_dataset("chain", 
                (nwalkers, N, l_var),
                dtype=np.float64)
            lp_ds = g.create_dataset("lnprob", 
                (nwalkers, N),
                dtype=np.float64)

        #args = [M.nplanets,M.rho_0,M.rho_0_unc,M.rho_prior,
        #    M.Teff,M.Teff_unc,M.logg,M.logg_unc,M.FeH,M.FeH_unc,    
        #    M.flux,M.err,M.fixed_sol,M.time,M._itime,M._ntt,
        #    M._tobs,M._omc,M._datatype,M.n_ldparams,M.ldfileloc,
        #    M.onlytransits,M.tregion]

        args = [M.nplanets,M.rho_0,M.rho_0_unc,M.rho_prior,
            M.ld1,M.ld1_unc,M.ld2,M.ld2_unc,M.ldp_prior,
            M.flux,M.err,M.fixed_sol,M.time,M._itime,M._ntt,
            M._tobs,M._omc,M._datatype,M.n_ldparams,M.ldfileloc,
            M.onlytransits,M.tregion]

        tom = tmod.logchi2_fitldp
        if runmpi:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom, 
                args=args,pool=pool)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom, 
                args=args,threads=th)

        time1 = thetime.time()
        p2, prob, state = sampler.run_mcmc(p0, burnin,
            storechain=False)
        sampler.reset()
        with h5py.File(outfile, u"a") as f:
            g = f["mcmc"]
            g.create_dataset("burnin_pos", data=p2)
            g.create_dataset("burnin_prob", data=prob)


        time2 = thetime.time()
        print 'burn-in took ' + str((time2 - time1)/60.) + ' min'
        time1 = thetime.time()
        for i, (pos, lnprob, state) in enumerate(sampler.sample(p2, 
            iterations=fullrun, rstate0=state,
            storechain=False)):

            #do the thinning in the loop here
            if i % thin == 0:
                ind = i / thin
                with h5py.File(outfile, u"a") as f:
                    g = f["mcmc"]
                    c_ds = g["chain"]
                    lp_ds = g["lnprob"]
                    c_ds[:, ind, :] = pos
                    lp_ds[:, ind] = lnprob

        time2 = thetime.time()
        print 'MCMC run took ' + str((time2 - time1)/60.) + ' min'
        print
        print("Mean acceptance: "
            + str(np.mean(sampler.acceptance_fraction)))
        print

        #try:
        #    print("Autocorrelation time:", sampler.acor)
        #    print("Autocorrelation times sampled:", 
        #        fullrun / sampler.acor)
        #except RuntimeError:
        #    print("No Autocorrelation")

        if runmpi:
            pool.close()
        # if doplot:
        #     plt.ioff()
        #     import triangle
        #     labels=[r"rho", r"zpt"]
        #     for ij in xrange(dat_d['nplanets']):
        #         labels = np.r_[labels,[r"T0",
        #             r"per",r"b", r"rprs", r"ecosw",r"esinw"]]
        #     figure = triangle.corner(sampler.flatchain, labels=labels)
        #     figure.savefig("data.png")

        #savefile = 'koi%s_np%s_prior%s.dat' %(dat_d['koi'],
        #    dat_d['nplanets'],rho_prior)
        #savefile2 = 'koi%s_np%s_prior%s_prob.dat' %(dat_d['koi'],
        #    dat_d['nplanets'],rho_prior)
        #np.savetxt(savefile,sampler.flatchain)
        #np.savetxt(savefile2,sampler.flatlnprobability)
        return sampler

if __name__ == "__main__":
    main()
