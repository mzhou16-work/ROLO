import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from ROLO import *

spec_dir  = '/Dedicated/jwang-data2/mzhou/unl-vrtm/data/Solar/'
light_Dir = '/Users/mzhou16/project/LightSpectrum/SpectrumData/'
file_spec = 'ASTM2000.dat'
sensor_dir = '/Users/mzhou16/project/OPNL_FILDA/FIRE_DETECTION_RESEARCH_CODE/PLANCK_FUNCTION/data/'

#------------------------------
# Read the sensor information...
one_band = pd.read_csv(sensor_dir + 'M11.csv')

#------------------------------
# Read the solar spectrum...
solar_spec = np.genfromtxt (spec_dir + file_spec, skip_header=1)
# get the wavelength in um 
solar_lamda = solar_spec[:,0]
# get the irradiance
solar_irrad = solar_spec[:,1]

# interpolate to M11
f = interpolate.interp1d(solar_lamda, solar_irrad)
solar_irrad_M11 = f(one_band['lambda'].values)
# solar_irrad_dnb = np.reshape(solar_irrad_dnb, (-1, 1))

#------------------------------
mpa = 0
date = '2024-02-24'
ideal = True

mpas = np.arange(-180, 180.1, 1)

irradiance_M11 = []

for mpa in mpas:
	wl, ak = get_lunar_albedo(date, mpa, KERNELS_PATH = './KERNELS/meta.tm', ideal = False)
	wl = wl/1000
	
	f_ak = interpolate.interp1d(wl, ak)
	ak_M11 = f_ak(one_band['lambda'].values)
	
	# in W/m2/micron
	lunar_irradiance_M11 = ak_M11 * solar_irrad_M11 * 6.4177*10**-5 /np.pi
# 	print(lunar_irradiance_M11)
# 	print( np.trapz(one_band['rsr'].values, one_band['lambda'].values, axis = 0), 
# 	       np.trapz(np.ones_like(one_band['lambda'].values), one_band['lambda'].values, axis = 0), 
# 	       one_band['lambda'].values[-1] - one_band['lambda'].values[0])
	# one_band['rsr'].values * lunar_irradiance_M11
	irrad_band   = np.trapz(one_band['rsr'].values * lunar_irradiance_M11, one_band['lambda'].values, axis = 0) / \
	               np.trapz(one_band['rsr'].values, one_band['lambda'].values, axis = 0)
	irradiance_M11.append(irrad_band)
	
irradiance_M11 = np.array(irradiance_M11)

fig = plt.figure(figsize = (6,4))
ax1   = fig.add_axes([0.2,0.2,0.6,0.6])
ax1.plot(mpas, irradiance_M11, color = 'k')
ax1.set_xlabel('Moonphase Angle')
ax1.set_ylabel('Irradiance (W$\cdot$m$^{-2}\cdot$µm$^{-1}$)')	
plt.savefig('./FIG/ROLO_spectrum_Reflectance_convolved_M11.png', dpi = 300)




