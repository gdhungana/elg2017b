import numpy as np
import desimodel.io
from astropy.io import fits

def sample_seeing(expfile,nobs=5000,seed=1234):
    rst=np.random.RandomState(seed)
    exp=fits.open(expfile)
    seeing=exp[1].data['seeing']
    sample=rst.choice(seeing,size=nobs)
    return sample

def sample_transparency(expfile,nobs=5000,seed=1234):
    rst=np.random.RandomState(seed)
    exp=fits.open(expfile)
    transparency=exp[1].data['transparency']
    sample=rst.choice(transparency,size=nobs)
    return sample

def sample_airmass(expfile,nobs=5000,seed=1234):
    rst=np.random.RandomState(seed)
    exp=fits.open(expfile)
    airmass=exp[1].data['airmass']
    sample=rst.choice(airmass,size=nobs)
    return sample


def sample_ebv(expfile,nobs=5000,seed=1234):
    #- sample the tileid, and get the E(B-V) from desi-tiles.fits file
    import os
    rst=np.random.RandomState(seed)
    exp=fits.open(expfile)
    tileid=exp[1].data['tileid']
    ii=rst.choice(len(tileid),size=nobs)
    sample_tiles=tileid[ii]

    #- find the match in desitiles 
    desitiles=fits.open(os.environ["DESIMODEL"]+'/data/footprint/desi-tiles.fits')
    dtileid=desitiles[1].data['TILEID']
    match=np.in1d(dtileid,sample_tiles)
    ebv=desitiles[1].data['EBV_MED'][match]

    return ebv


def sample_moon(expfile,nobs=5000,seed=1234):
    #- only for gray time
    rst=np.random.RandomState(seed)
    exp=fits.open(expfile)
    pas=exp[1].data['pass']
    #nexp=len(exp[1].data['moonfrac'])
    moonfrac=exp[1].data['moonfrac'][pas==4]
    moonsep=exp[1].data['moonsep'][pas==4]
    moonalt=exp[1].data['moonalt'][pas==4]
    #- sample with no correlation
    samp_frac=rst.choice(moonfrac,size=nobs)
    samp_sep=rst.choice(moonsep,size=nobs)
    samp_alt=rst.choice(moonalt,size=nobs)
    
    return samp_frac,samp_sep,samp_alt
    
