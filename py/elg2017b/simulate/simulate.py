from __future__ import print_function
import numpy as np
from astropy.table import Table, Column, vstack
from astropy.io import fits
import desisim
import desisim.templates
from desisim.obs import get_night
from desispec.io import empty_fibermap, Brick,fitsheader
from desispec.resolution import Resolution
import desiutil.io
import astropy.units as u
from specsim import simulator
import sys,os
import desimodel.io

def get_templates(wavelength,seed=1234,nmodel=5000):
    elg=desisim.templates.ELG(wave=wavelength)
    flux,tmpwave,meta1=elg.make_templates(nmodel=nmodel,seed=seed)
    return flux,tmpwave,meta1

def update_simulator(qsim, exptime=None, airmass=None, seeing=None, moon_frac=None, moon_sep=None, moon_alt=None, galsim=False):

    if airmass is not None:
        qsim.atmosphere.airmass=airmass
    else:
        airmass=qsim.atmosphere.airmass

    if seeing is not None:
        qsim.atmosphere.seeing_fwhm_ref=seeing*u.arcsec
    else:
        seeing=qsim.atmosphere.seeing_fwhm_ref.value

    if galsim:
        qsim.instrument.fiberloss_method='galsim'

    if moon_frac is not None:
        qsim.atmosphere.moon.moon_phase=np.arccos(2 * moon_frac - 1) / np.pi
    else:
        moon_frac=0.5

    if moon_sep is not None:
        qsim.atmosphere.moon.separation_angle=moon_sep*u.deg
    else:
        moon_sep=60.

    if moon_alt is not None:
        qsim.atmosphere.moon.moon_zenith=(90-moon_alt)*u.deg
    else:
        moon_alt=-10.
     
    if exptime is not None:
        qsim.observation.exposure_time=exptime*u.s

    else:
        #- fixing transparency to 1.
        exptime=get_exptime(seeing,1.0,airmass,ebv,moon_frac,moon_sep,moon_alt,program=qsim.atmosphere.condition)

    return qsim

def simulate(airmass=None,exptime=None,seeing=None,moon_frac=None,moon_sep=None,moon_alt=None,seed=1234,nspec=5000,
brickname='testbrick',galsim=False,ra=None,dec=None):

    #- construct the simulator
    qsim=simulator.Simulator('desi')

    # Initialize random number generator to use.
    random_state = np.random.RandomState(seed)

    #- Create a blank fake fibermap for bricks
    fibermap = empty_fibermap(nspec)
    targetids = random_state.randint(2**62, size=nspec)
    fibermap['TARGETID'] = targetids
    night = get_night()
    expid = 0

    #- working out only ELG
    objtype='ELG'
    true_objtype = np.tile(np.array([objtype]),(nspec))

    #- Initialize the output truth table.
    spectra = []
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
    dw = 0.2
    wavelengths = np.arange(round(wavemin, 1), wavemax, dw)

    npix = len(wavelengths)
    truth = dict()
    meta = Table()
    truth['OBJTYPE'] = 'ELG'*nspec
    truth['FLUX'] = np.zeros((nspec, npix))
    truth['WAVE'] = wavelengths

    #- get the templates
    flux,tmpwave,meta1=get_templates(wavelengths,seed=seed,nmodel=nspec)
    truth['FLUX']=flux
    meta=vstack([meta,meta1])
    

    #- Add TARGETID and the true OBJTYPE to the metadata table.
    meta.add_column(Column(true_objtype, dtype=(str, 10), name='TRUE_OBJTYPE'))
    meta.add_column(Column(targetids, name='TARGETID'))

    #- Rename REDSHIFT -> TRUEZ anticipating later table joins with zbest.Z
    meta.rename_column('REDSHIFT', 'TRUEZ')

    waves, trueflux, noisyflux, obsivar, resolution, sflux = {}, {}, {}, {}, {}, {}
    
    #- Now simulate
    maxbin = 0
    nmax= nspec
    for camera in qsim.instrument.cameras:
        # Lookup this camera's resolution matrix and convert to the sparse
        # format used in desispec.
        R = Resolution(camera.get_output_resolution_matrix())
        resolution[camera.name] = np.tile(R.to_fits_array(), [nspec, 1, 1])
        waves[camera.name] = (camera.output_wavelength.to(u.Angstrom).value.astype(np.float32))
        nwave = len(waves[camera.name])
        maxbin = max(maxbin, len(waves[camera.name]))
        nobj = np.zeros((nmax,3,maxbin)) # object photons
        nsky = np.zeros((nmax,3,maxbin)) # sky photons
        nivar = np.zeros((nmax,3,maxbin)) # inverse variance (object+sky)
        cframe_observedflux = np.zeros((nmax,3,maxbin))  # calibrated object flux
        cframe_ivar = np.zeros((nmax,3,maxbin)) # inverse variance of calibrated object flux
        cframe_rand_noise = np.zeros((nmax,3,maxbin)) # random Gaussian noise to calibrated flux
        sky_ivar = np.zeros((nmax,3,maxbin)) # inverse variance of sky
        sky_rand_noise = np.zeros((nmax,3,maxbin)) # random Gaussian noise to sky only
        frame_rand_noise = np.zeros((nmax,3,maxbin)) # random Gaussian noise to nobj+nsky
        trueflux[camera.name] = np.empty((nspec, nwave)) # calibrated brick flux
        noisyflux[camera.name] = np.empty((nspec, nwave)) # brick flux with noise
        obsivar[camera.name] = np.empty((nspec, nwave)) # inverse variance of brick flux

        sflux = np.empty((nspec, npix))

    #- Repeat the simulation for all spectra
    fluxunits = 1e-17 * u.erg / (u.s * u.cm ** 2 * u.Angstrom)
    spectra=truth['FLUX']*1.0e17
    print("Simulating Spectra")
    for j in range(nspec):
        print("Simulating %s/%s spectra"%(j,nspec),end='\r')
        thisobjtype='ELG'
        sys.stdout.flush()

        #- update qsim using conditions
        if airmass is None:
            thisairmass=None
        else: thisairmass=airmass[j]

        if seeing is None:
            thisseeing=None
        else: thisseeing=seeing[j]

        if moon_frac is None:
            thismoon_frac=None
        else: thismoon_frac=moon_frac[j]

        if moon_sep is None:
            thismoon_sep=None
        else: thismoon_sep=moon_sep[j]
        
        if moon_alt is None:
            thismoon_alt=None
        else: thismoon_alt=moon_alt[j]
        
        if exptime is None:
            thisexptime=None

        else: thisexptime=exptime[j]

        nqsim=update_simulator(qsim,airmass=thisairmass, exptime=thisexptime, seeing=thisseeing, moon_frac=thismoon_frac, moon_sep=thismoon_sep, moon_alt=thismoon_alt,galsim=galsim)

        nqsim.source.update_in(
                'Quickgen source {0}'.format(j), thisobjtype.lower(),
                wavelengths * u.Angstrom, spectra[j, :] * fluxunits)
        nqsim.source.update_out()

        nqsim.simulate()
        nqsim.generate_random_noise(random_state)

        sflux[j][:] = 1e17 * qsim.source.flux_in.to(fluxunits).value


        for i, output in enumerate(nqsim.camera_output):
            assert output['observed_flux'].unit == 1e17 * fluxunits
            # Extract the simulation results needed to create our uncalibrated
            # frame output file.
            num_pixels = len(output)
            nobj[j, i, :num_pixels] = output['num_source_electrons'][:,0]
            nsky[j, i, :num_pixels] = output['num_sky_electrons'][:,0]
            nivar[j, i, :num_pixels] = 1.0 / output['variance_electrons'][:,0]

            # Get results for our flux-calibrated output file.
            cframe_observedflux[j, i, :num_pixels] = 1e17 * output['observed_flux'][:,0]
            cframe_ivar[j, i, :num_pixels] = 1e-34 * output['flux_inverse_variance'][:,0]

            # Fill brick arrays from the results.
            camera = output.meta['name']
            trueflux[camera][j][:] = 1e17 * output['observed_flux'][:,0]
            noisyflux[camera][j][:] = 1e17 * (output['observed_flux'][:,0] +
                output['flux_calibration'][:,0] * output['random_noise_electrons'][:,0])
            #return output
            obsivar[camera][j][:] = 1e-34 * output['flux_inverse_variance'][:,0]

            # Use the same noise realization in the cframe and frame, without any
            # additional noise from sky subtraction for now.
            frame_rand_noise[j, i, :num_pixels] = output['random_noise_electrons'][:,0]
            cframe_rand_noise[j, i, :num_pixels] = 1e17 * (
                output['flux_calibration'][:,0] * output['random_noise_electrons'][:,0])

            # The sky output file represents a model fit to ~40 sky fibers.
            # We reduce the variance by a factor of 25 to account for this and
            # give the sky an independent (Gaussian) noise realization.
            sky_ivar[j, i, :num_pixels] = 25.0 / (
                output['variance_electrons'][:,0] - output['num_source_electrons'][:,0])
            sky_rand_noise[j, i, :num_pixels] = random_state.normal(
                scale=1.0 / np.sqrt(sky_ivar[j,i,:num_pixels]),size=num_pixels)
            cframe_flux=cframe_observedflux[j,i,:num_pixels]+cframe_rand_noise[j,i,:num_pixels]
 
    armName={"b":0,"r":1,"z":2}
    for channel in 'brz':

        num_pixels = len(waves[channel])
        dwave=np.gradient(waves[channel])
        nobj[:,armName[channel],:num_pixels]/=dwave
        frame_rand_noise[:,armName[channel],:num_pixels]/=dwave
        nivar[:,armName[channel],:num_pixels]*=dwave**2
        nsky[:,armName[channel],:num_pixels]/=dwave
        sky_rand_noise[:,armName[channel],:num_pixels]/=dwave
        sky_ivar[:,armName[channel],:num_pixels]/=dwave**2

        # Now write the outputs in DESI standard file system. None of the output file can have more than 500 spectra

        # Output brick files
        if ra is None or dec is None:
            filename = 'brick-{}-{}.fits'.format(channel, brickname)
            filepath = os.path.normpath(os.path.join('{}'.format(brickname),filename))
            if os.path.exists(filepath):
                os.remove(filepath)
            print('Writing {}'.format(filepath))

            header = dict(BRICKNAM=brickname, CHANNEL=channel)
            brick = Brick(filepath, mode='update', header=header)
            brick.add_objects(noisyflux[channel], obsivar[channel],
                    waves[channel], resolution[channel], fibermap, night, expid)
            brick.close()
            """
            # Append truth to the file. Note: we add the resolution-convolved true
            # flux, not the high resolution source flux, which makes chi2
            # calculations easier.
            header = fitsheader(header)
            fx = fits.open(filepath, mode='append')
            _add_truth(fx, header, meta, trueflux, sflux, wavelengths, channel)
            fx.flush()
            fx.close()
            #sys.stdout.close()
            """
            print ("Wrote file {}".format(filepath))
        else:
            bricknames=get_bricknames(ra,dec)
            fibermap['BRICKNAME']=bricknames
            bricknames=set(bricknames)
            print("No. of bricks: {}".format(len(bricknames)))
            print ("Writing brick files")
            for brick_name in bricknames:

                thisbrick=(fibermap['BRICKNAME']==brick_name)
                brickdata=fibermap[thisbrick]
                
                fibers=brickdata['FIBER']#np.mod(brickdata['FIBER'],nspec)
                filename= 'brick-{}-{}.fits'.format(channel,brick_name)
                filepath= os.path.normpath(os.path.join('./{}'.format(brick_name),filename))
                if os.path.exists(filepath):
                    os.remove(filepath)
                #print('Writing {}'.format(filepath))

                header=dict(BRICKNAM=brick_name,CHANNEL=channel)
                brick=Brick(filepath,mode='update',header=header)
                brick.add_objects(noisyflux[channel][fibers], obsivar[channel][fibers], waves[channel],resolution[channel][fibers],brickdata,night,expid)
                brick.close()
            print("Finished writing brick files for {} bricks".format(len(bricknames)))
    #- make a truth file
    header=fitsheader(header)
    make_truthfile(header,meta,trueflux,sflux,wavelengths)
   # return waves, trueflux, noisyflux, obsivar, sflux

def _add_truth(hdus, header, meta, trueflux, sflux, wave, channel):
    """Utility function for adding truth to an output FITS file."""
    hdus.append(
        fits.ImageHDU(trueflux[channel], name='_TRUEFLUX', header=header))
    if channel == 'b':
        swave = wave.astype(np.float32)
        hdus.append(fits.ImageHDU(swave, name='_SOURCEWAVE', header=header))
        hdus.append(fits.ImageHDU(sflux, name='_SOURCEFLUX', header=header))
        metatable = desiutil.io.encode_table(meta, encoding='ascii')
        metahdu = fits.convenience.table_to_hdu(meta)
        metahdu.header['EXTNAME'] = '_TRUTH'
        hdus.append(metahdu)

def get_bricknames(ra,dec):
    bricknames=brick.brickname(ra,dec)
    return bricknames

def make_truthfile(header,meta,trueflux,sflux,wave):
    """Utility function for adding truth to an output FITS file."""
    truthfile='truth.fits'
    truthpath='./'+truthfile
    #if os.path.exists(truthpath):
    #    os.remove(truthpath)
    print('Writing {}'.format(truthpath))
    hx=fits.HDUList([fits.PrimaryHDU(header=header)])
    hx.append(fits.ImageHDU(trueflux['r'], name='_TRUEFLUX', header=header))
    swave = wave.astype(np.float32)
    hx.append(fits.ImageHDU(swave, name='_SOURCEWAVE', header=header))
    hx.append(fits.ImageHDU(sflux, name='_SOURCEFLUX', header=header))
    metatable = desiutil.io.encode_table(meta, encoding='ascii')
    metahdu = fits.convenience.table_to_hdu(meta)
    metahdu.header['EXTNAME'] = '_TRUTH'
    hx.append(metahdu)
    #hx.flush()
    #hdus.close()
    hx.writeto(truthpath,clobber=True)
    print ("Wrote truth file {}".format(truthpath))
