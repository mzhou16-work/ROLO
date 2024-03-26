import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from ROLO import *

spec_dir  = '/Dedicated/jwang-data2/mzhou/unl-vrtm/data/Solar/'
light_Dir = '/Users/mzhou16/project/LightSpectrum/SpectrumData/'
file_spec = 'ASTM2000.dat'

sensor_dir = '/Users/mzhou16/project/OPNL_FILDA/FIRE_DETECTION_RESEARCH_CODE/PLANCK_FUNCTION/data/'

one_band = pd.read_csv(sensor_dir + 'M11.csv')

#------------------------------
abs_mpa_rad = 0
date = '2024-02-24'
ideal = True

wl, ak = get_lunar_albedo(date, abs_mpa_rad, KERNELS_PATH = './KERNELS/meta.tm', ideal = False)
wl = wl/1000

f_ak = interpolate.interp1d(wl, ak)
ak_M11 = f_ak(one_band['lambda'].values)

print(ak_M11.shape)

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

# in nW/cm2/micron
lunar_irradiance_M11 = ak_M11 * solar_irrad_M11 * 6.4177*10**-5 /np.pi

fig = plt.figure(figsize = (6,4))
ax1   = fig.add_axes([0.2,0.2,0.6,0.6])
lns1 = ax1.plot(one_band['lambda'].values, ak_M11, color = 'C1', ls = '--', label = 'Lunar Disk Reflectance')
ax1.set_xlabel('Wavelength (µm)')
ax1.set_ylabel('Disk Reflectance')
ax1.set_ylim(0.01, np.nanmax(ak)*1.2)
lns2 = ax1.fill_between(one_band['lambda'].values, 0, one_band['rsr'].values * np.max(ak_M11), color = 'grey', alpha = 0.4, label = 'DNB RSR')

ax2 = ax1.twinx()
lns3 = ax2.plot(one_band['lambda'].values, lunar_irradiance_M11, label = 'Lunar Irradiance')
ax2.set_ylabel('Irradiance (W$\cdot$m$^{-2}\cdot$µm$^{-1}$)')
ax2.set_ylim(0.0005, 0.00065)

lns = lns1 + lns3 

lns.append(lns2)
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc="lower left",ncol=1, fontsize=12, frameon=False)

plt.savefig('./FIG/ROLO_spectrum_Reflectance_M11.png', dpi = 300)
