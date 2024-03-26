import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from ROLO import *
from pylib.AERML_VIIRS.AERML_IO import *
from pylib.AERML_VIIRS.AMRML_ProSat import *
from pylib.AERML_VIIRS.AERML_celestial import *
from pylib.packages_basic import *
from pylib.packages_viirs import *

spec_dir  = '/Dedicated/jwang-data2/mzhou/unl-vrtm/data/Solar/'
light_Dir = '/Users/mzhou16/project/LightSpectrum/SpectrumData/'
file_spec = 'ASTM2000.dat'

lamdas = np.arange(350, 1101, 1)/1000

input_theta = [5, 15, 30, 45]

print('input_theta', input_theta)
#-----------------------------------------------------------------------
# Read the solar spectrum...
solar_spec = np.genfromtxt (spec_dir + file_spec, skip_header=1)
# get the wavelength
solar_lamda = solar_spec[:,0]
# get the irradiance
solar_irrad = solar_spec[:,1]
# interpolate the solar spectrum
f = interpolate.interp1d(solar_lamda, solar_irrad)
solar_irrad = f(lamdas)
solar_irrad = np.reshape(solar_irrad, (-1, 1))

#-----------------------------------------------------------------------

lunar_irrad_miller = []
for theta in input_theta:
	lunar = lunar_model(lamdas, theta)
	lunar_spec = lunar.alphas * solar_irrad * lunar.F * lunar.f_theta * 1000
	lunar_irrad_miller.append(lunar_spec)



#------------------------------
abs_mpa_rad = 0
date = '2024-02-24'
ideal = True

lunar_irrad_ROLO = []
for theta in input_theta:

	wl, ak = get_lunar_albedo(date, theta, KERNELS_PATH = './KERNELS/meta.tm', ideal = False)
	wl = wl/1000
	f_ak = interpolate.interp1d(wl, ak)

	ak_oneband = np.reshape(f_ak(lamdas), (-1, 1))
	
# 	# in W/m2/micron/sr
	lunar_irrad = ak_oneband * solar_irrad * 6.4177*10**-5 /np.pi*1000

	lunar_irrad_ROLO.append( lunar_irrad )


fig, ax = def_figure1D(figuresize = (7, 5))
for i in range(len(lunar_irrad_miller)):
	ax.plot(lamdas, lunar_irrad_miller[i], zorder = 2, color = 'C' + str(i), ls = '-', label = 'M&T, MPA ' + str(input_theta[i]) + ' deg')
	ax.plot(lamdas, lunar_irrad_ROLO[i], zorder = 2, color = 'C' + str(i),  ls = '--', label = 'ROLO, MPA ' + str(input_theta[i]) + ' deg')
ax.legend(frameon = False, loc = 'lower center', ncol=3, bbox_to_anchor=(0.5, -0.36))
ax.set_xlabel('wavelength')
ax.set_ylabel('Irradiance (W$\cdot$m$^{-2}\cdot$µm$^{-1}$)')
ax.set_title('Fire pixel')
plt.savefig('./FIG/ROLO_MILLER_spectrum_Reflectance.png', dpi = 300)
































