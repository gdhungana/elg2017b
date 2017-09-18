from desisurvey import etc
from desisurvey import utils as utils
import numpy as np

def get_exptime(seeing,transparency,airmass,ebv,moon_frac,moon_sep,moon_alt,program='DARK'):

    value=etc.exposure_time(program,seeing,transparency,airmass,ebv,moon_frac,moon_sep,moon_alt)
    exp=value.value
    return np.clip(exp,0.,3600.)
    
    
