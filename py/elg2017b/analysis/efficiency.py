from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import scipy.special as sp

#- redshift errors and zwarn fractions from DESI-1657
#- sigmav = c sigmaz / (1+z)
_sigma_v = {
    'ELG': 19.,
    'LRG': 40.,
    'BGS': 13.,
    'QSO': 423.,
    'STAR': 18.,
    'SKY': 9999,      #- meaningless
    'UNKNOWN': 9999,  #- meaningless
}

_cata_fail_fraction = {
   # Catastrophic error fractions from redmonster on oak (ELG, LRG, QSO)
    'ELG': 0.08,
    'LRG': 0.013,
    'QSO': 0.20,
    'BGS': 0.,
    'STAR': 0.,
    'SKY': 0.,
    'UNKNOWN': 0.,
}

def get_zeff_obs_newmodel(simtype, obsconditions):
    #- only model for 'ELG'
    if(simtype=='ELG'):
        p_v = [0.86, 0.11, -0.02]
        p_w = [0.86, 0.92, 0.62]
        p_x = [0.86, -0.10, -0.01]
        p_y = [0.86, -0.15, -0.20]
        p_z = [0.86, -10.0, 300.0]
        sigma_r = 0.071 #- adding all the delta eff residuals in quad.
    
    else:
        log.warning('No model for how observing conditions impact {} redshift efficiency'.format(simtype))
        return np.ones(len(obsconditions))
    # airmass
    v = obsconditions['AIRMASS'] - 1.0
    pv  = p_v[0] + p_v[1] * v + p_v[2] * (v**2. - np.mean(v**2))

    # ebmv
    w = obsconditions['EBMV'] - 0.0
    pw = p_w[0] + p_w[1] * w + p_w[2] * (w**2 - np.mean(w**2))

    # seeing
    x = obsconditions['SEEING'] - 1.1
    px = p_x[0] + p_x[1]*x + p_x[2] * (x**2 - np.mean(x**2))

    # transparency
    y = obsconditions['LINTRANS'] - 1.0
    py = p_y[0] + p_y[1]*y + p_y[2] * (y**2 - np.mean(y**2))

    pr = 1.0 + np.random.normal(size=len(y), scale=sigma_r)

    pobs = (pv * pw * px * py * pr).clip(min=0.0)
    return pobs

def get_zeff_obs(simtype, obsconditions):
    '''
    '''
    if(simtype=='LRG'):
        p_v = [1.0, 0.15, -0.5]
        p_w = [1.0, 0.4, 0.0]
        p_x = [1.0, 0.06, 0.05]
        p_y = [1.0, 0.0, 0.08]
        p_z = [1.0, 0.0, 0.0]
        sigma_r = 0.02
    elif(simtype=='QSO'):
        p_v = [1.0, -0.2, 0.3]
        p_w = [1.0, -0.5, 0.6]
        p_x = [1.0, -0.1, -0.075]
        p_y = [1.0, -0.08, -0.04]
        p_z = [1.0, 0.0, 0.0]
        sigma_r = 0.05
    elif(simtype=='ELG'):
        p_v = [1.0, -0.1, -0.2]
        p_w = [1.0, 0.25, -0.75]
        p_x = [1.0, 0.0, 0.05]
        p_y = [1.0, 0.2, 0.1]
        p_z = [1.0, -10.0, 300.0]
        sigma_r = 0.075
    else:
        log.warning('No model for how observing conditions impact {} redshift efficiency'.format(simtype))
        return np.ones(len(obsconditions))

    # airmass
    v = obsconditions['AIRMASS'] - np.mean(obsconditions['AIRMASS'])
    pv  = p_v[0] + p_v[1] * v + p_v[2] * (v**2. - np.mean(v**2))

    # ebmv
    w = obsconditions['EBMV'] - np.mean(obsconditions['EBMV'])
    pw = p_w[0] + p_w[1] * w + p_w[2] * (w**2 - np.mean(w**2))

    # seeing
    x = obsconditions['SEEING'] - np.mean(obsconditions['SEEING'])
    px = p_x[0] + p_x[1]*x + p_x[2] * (x**2 - np.mean(x**2))

    # transparency
    y = obsconditions['LINTRANS'] - np.mean(obsconditions['LINTRANS'])
    py = p_y[0] + p_y[1]*y + p_y[2] * (y**2 - np.mean(y**2))

    # moon illumination fraction
    #z = obsconditions['MOONFRAC'] - np.mean(obsconditions['MOONFRAC'])
    #pz = p_z[0] + p_z[1]*z + p_z[2] * (z**2 - np.mean(z**2))

    #- if moon is down phase doesn't matter
    #pz[obsconditions['MOONALT'] < 0] = 1.0

    pr = 1.0 + np.random.normal(size=len(y), scale=sigma_r)

    #- this correction factor can be greater than 1, but not less than 0
    pobs = (pv * pw * px * py * pr).clip(min=0.0)

    return pobs


def get_redshift_efficiency(simtype, catfile,plot=False, newmodel=False):
    """
    Simple model to get the redshift effiency from the observational conditions or observed magnitudes+redshuft
    Args:
        simtype: ELG, LRG, QSO, MWS, BGS
        targets: target catalog table; currently used only for TARGETID
        truth: truth table with OIIFLUX, TRUEZ
        targets_in_tile: dictionary. Keys correspond to tileids, its values are the
            arrays of targetids observed in that tile.
        obsconditions: table observing conditions with columns
           'TILEID': array of tile IDs
           'AIRMASS': array of airmass values on a tile
           'EBMV': array of E(B-V) values on a tile
           'LINTRANS': array of atmospheric transparency during spectro obs; floats [0-1]
           'MOONFRAC': array of moonfraction values on a tile.
           'SEEING': array of FWHM seeing during spectroscopic observation on a tile.
    Returns:
        tuple of arrays (observed, p) both with same length as targets
        observed: boolean array of whether the target was observed in these tiles
        p: probability to get this redshift right
    """

    catalog=fits.open(catfile)
    datacat=catalog[1].data
    n=len(datacat)

    obsconditions={'AIRMASS': datacat['AIRMASS'], 'SEEING': datacat['SEEING'], 'LINTRANS': datacat['TRANSPARENCY'], 'EBMV': np.zeros(n)}#datacat['EBMV']}

    if (simtype == 'ELG'):
        # Read the model OII flux threshold (FDR fig 7.12 modified to fit redmonster efficiency on OAK)
        filename = ('./quickcat_elg_oii_flux_threshold.txt')
        fdr_z, modified_fdr_oii_flux_threshold = np.loadtxt(filename, unpack=True)

        # Get OIIflux from truth
        true_oii_flux = datacat['OIIFLUX']

        # Compute OII flux thresholds for truez
        oii_flux_threshold = np.interp(datacat['Z'],fdr_z,modified_fdr_oii_flux_threshold)
        assert (oii_flux_threshold.size == true_oii_flux.size),"oii_flux_threshold and true_oii_flux should have the same size"
        if plot:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            selz=np.where((fdr_z>= 0.6) & (fdr_z<=1.6))[0]
            ax.plot(fdr_z[selz],modified_fdr_oii_flux_threshold[selz],'r-',label='[OII] flux threshold',lw=2)
            ax.legend(fontsize=22,numpoints=1)
            ax.set_xlabel(r'$z$',fontsize=20)
            ax.set_xlim(0.6,1.6)
            ax.set_ylabel(r'${\rm [OII]\ flux\ [ergs/s/cm^2]}$',fontsize=20)
            ax.tick_params('both',labelsize=16)
            plt.show()

        # efficiency is modeled as a function of flux_OII/f_OII_threshold(z) and an arbitrary sigma_fudge
        sigma_fudge = 1.0
        max_efficiency = 1.0
        simulated_eff = eff_model(true_oii_flux/oii_flux_threshold,sigma_fudge,max_efficiency)

    else:
        default_zeff = 0.98
        log.warning('using default redshift efficiency of {} for {}'.format(default_zeff, simtype))
        simulated_eff = default_zeff * np.ones(n)

    if newmodel:
        print("using new efficiency model instead of original")
        zeff_obs = get_zeff_obs_newmodel(simtype,obsconditions)
    else:
        print("Using original quickcat efficiency model")
        zeff_obs = get_zeff_obs(simtype, obsconditions)
    pfail = np.ones(n)
    #observed = np.zeros(n, dtype=bool)

    tmp=(simulated_eff*zeff_obs).clip(0,1)
    pfail*=(1-tmp)
    simulated_eff=(1-pfail)
    return simulated_eff

# Efficiency model
def eff_model(x, sigma, max_efficiency):
    return 0.5*max_efficiency*(1.+sp.erf((x-1)/(np.sqrt(2.)*sigma)))
    


def get_observed_redshifts(catfile,newmodel=False):
    """
    Returns observed z, zerr, zwarn arrays given true object types and redshifts
    Args:
        targets: target catalog table; currently used only for target mask bits
        truth: truth table with OIIFLUX, TRUEZ
        targets_in_tile: dictionary. Keys correspond to tileids, its values are the
            arrays of targetids observed in that tile.
        obsconditions: table observing conditions with columns
           'TILEID': array of tile IDs
           'AIRMASS': array of airmass values on a tile
           'EBMV': array of E(B-V) values on a tile
           'LINTRANS': array of atmospheric transparency during spectro obs; floats [0-1]
           'MOONFRAC': array of moonfraction values on a tile.
           'SEEING': array of FWHM seeing during spectroscopic observation on a tile.
    Returns:
        tuple of (zout, zerr, zwarn)
    """

    
    catalog=fits.open(catfile)
    datacat=catalog[1].data
    truez = datacat['Z']

    zout = truez.copy()
    zerr = np.zeros(len(truez), dtype=np.float32)
    zwarn = np.zeros(len(truez), dtype=np.int32)
    simtype = np.array(['ELG']*len(truez))
    objtypes = ['ELG']
    print len(objtypes)

    for objtype in objtypes:
        if objtype in _sigma_v.keys():
            ii = np.where(simtype==objtype)[0]

            n = len(ii)#np.count_nonzero(ii)
            print n
            # Error model for ELGs
            if (objtype =='ELG'):
                filename = './quickcat_elg_oii_errz.txt'
                oii, errz_oii = np.loadtxt(filename, unpack=True)
                try:
                    true_oii_flux = datacat['OIIFLUX'][ii]
                except:
                    raise Exception('Missing OII flux information to estimate redshift error for ELGs')

                mean_err_oii = np.interp(true_oii_flux,oii,errz_oii)
                zerr[ii] = mean_err_oii*(1.+truez[ii])
                zout[ii] += np.random.normal(scale=zerr[ii])

            else:
                zerr[ii] = _sigma_v[objtype] * (1+truez[ii]) / c
                zout[ii] += np.random.normal(scale=zerr[ii])

            # Set ZWARN flags for some targets
            # the redshift efficiency only sets warning, but does not impact
            # the redshift value and its error.
            goodz_prob = get_redshift_efficiency(objtype, catfile, newmodel=newmodel)

            assert len(goodz_prob) == n
            r = np.random.random(n)
            zwarn[ii] = 4 * (r > goodz_prob)

            # Add fraction of catastrophic failures (zwarn=0 but wrong z)
            nzwarnzero = np.count_nonzero(zwarn[ii] == 0)
            num_cata = np.random.poisson(_cata_fail_fraction[objtype] * nzwarnzero)
            if (objtype == 'ELG'): zlim=[0.6,0.7]
            if num_cata > 0:
                #- tmp = boolean array for all targets, flagging only those
                #- that are of this simtype and were observed this epoch
                kk, = np.where((zwarn==0))
                index = np.random.choice(kk, size=num_cata, replace=False)
                assert np.all(np.in1d(index, np.where(ii)[0]))
                assert np.all(zwarn[index] == 0)

                zout[index] = np.random.uniform(zlim[0],zlim[1],len(index))

        else:
            msg = 'No redshift efficiency model for {}; using true z\n'.format(objtype) + \
                  'Known types are {}'.format(list(_sigma_v.keys()))
            print(msg)

    return zout, zerr, zwarn

def quickcat(catfile,outfile,newmodel=False):

    cat=fits.open(catfile)
    datacat=cat[1].data

    ra=datacat['RA']
    dec=datacat['DEC']
    
    zout,zerr,zwarn=get_observed_redshifts(catfile,newmodel=newmodel)

    kk=np.where(zwarn==0)[0]
    ra=ra[kk]
    dec=dec[kk]
    z=zout[kk]
    zwarn=zwarn[kk]
    zerr=zerr[kk]


    zcatalog=Table([ra,dec,z,zerr,zwarn],names=('RA','DEC','Z','ZERR','ZWARN'))
    zcatalog.write(outfile,format='fits',overwrite=True)
    

    
