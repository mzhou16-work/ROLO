import numpy as np
import spiceypy as spice

lamda = np.array([350.0, 355.1, 405.0, 412.3, 414.4, 441.6, 465.8, 475.0, 486.9, 
         544.0, 549.1, 553.8, 665.1, 693.1, 703.6, 745.3, 763.7, 774.8, 
         865.3, 872.6, 882.0, 928.4, 939.3, 942.1, 1059.5, 1243.2, 1538.7, 
         1633.6, 1981.5, 2126.3, 2250.9, 2383.6])

a_0 = np.array([-2.67511, -2.71924, -2.35754, -2.34185, -2.43367, -2.31964, 
	   -2.35085, -2.28999, -2.23351, -2.13864, -2.10782, -2.12504, 
	   -1.88914, -1.89410, -1.92103, -1.86896, -1.85258, -1.80271, 
	   -1.74561, -1.76779, -1.73011, -1.75981, -1.76245, -1.66473,
	   -1.59323, -1.53594, -1.33802, -1.34567, -1.26203, -1.18946, 
	   -1.04232, -1.08403])


a_1 = np.array([-1.78539, -1.74298, -1.72134, -1.74337, -1.72184, -1.72114, 
       -1.66538, -1.6318, -1.68573, -1.60613, -1.66736, -1.6597, 
       -1.58096, -1.58509, -1.60151, -1.57522, -1.47181, -1.59357, 
       -1.58482, -1.60345, -1.61156, -1.45395, -1.49892, -1.61875, 
       -1.71358, -1.55214, -1.46208, -1.46057, -1.25138, -2.55069, 
       -1.46809, -1.31032])


a_2 = np.array([0.50612, 0.44523, 0.40337, 0.42156, 0.43600, 0.37286, 0.41802, 
       0.36193, 0.37632, 0.27886, 0.41697, 0.38409, 0.30477, 0.2808, 
       0.36924, 0.33712, 0.14377, 0.36351, 0.35009, 0.37974, 0.36115, 
       0.13780, 0.07956, 0.14630, 0.50599, 0.31479, 0.15784, 0.23813, 
       -0.06569,2.10026, 0.43817, 0.20323])

a_3 = np.array([-0.25578, -0.23315, -0.21105, -0.21512, -0.22675, -0.19304, 
       -0.22541, -0.20381, -0.19877, -0.16426, -0.22026, -0.20655, 
       -0.17908, -0.16427, -0.20567, -0.19415, -0.11589, -0.20326, 
       -0.19569, -0.20625, -0.19576, -0.11254, -0.07546, -0.09216, 
       -0.25178, -0.18178, -0.11712, -0.15494, -0.04005, -0.87285, 
       -0.24632, -0.15863])

b_1 = np.array([0.03744, 0.03492, 0.03505, 0.03141, 0.03474, 0.03736, 0.04274, 
       0.04007, 0.03881, 0.03833, 0.03451, 0.04052, 0.04415, 0.04429, 
       0.04494, 0.03967, 0.04435, 0.04710, 0.04142, 0.04645, 0.04847, 
       0.05000, 0.05461, 0.04533, 0.04906, 0.03965, 0.04674, 0.03883, 
       0.04157, 0.03819, 0.04893, 0.05955])

b_2 = np.array([0.00981, 0.01142, 0.01043, 0.01364, 0.01188, 0.01545, 0.01127, 
       0.01216, 0.01566, 0.01189, 0.01452, 0.01009, 0.00983, 0.00914, 
       0.00987, 0.01318, 0.02000, 0.01196, 0.01612, 0.01170, 0.01065, 
       0.01476, 0.01355, 0.0301, 0.031780, 0.03009, 0.01471, 0.02280, 
       0.02036, -0.00685, 0.00617, -0.0094])

b_3 = np.array([-0.00322, -0.00383, -0.00341, -0.00472, -0.00422, -0.00559, 
       -0.00439, -0.00437, -0.00555, -0.00390, -0.00517, -0.00388, 
       -0.00389, -0.00351, -0.00386, -0.00464, -0.00738, -0.00476, 
       -0.00550, -0.00424, -0.00404, -0.00513, -0.00464, -0.01166, 
       -0.01138, -0.01123, -0.00656, -0.00877, -0.00772, -0.00200, 
       -0.00259, 0.00083])

d_1 = np.array([0.34185, 0.33875, 0.35235, 0.36591, 0.35558, 0.37935, 0.33450, 
       0.33024, 0.36590, 0.37190, 0.36814, 0.37206, 0.37141, 0.39109, 
       0.37155, 0.36888, 0.39126, 0.36908, 0.39200, 0.39354, 0.40714, 
       0.41900, 0.47936, 0.57275, 0.48160, 0.49040, 0.53831, 0.54393, 
       0.49099, 0.29239, 0.38154, 0.36134])

d_2 = np.array([ 0.01441, 0.01612, -0.03818, -0.05902, -0.03247, -0.09562, 
       -0.02546, -0.03131, -0.08945, -0.10629, -0.09815, -0.10745, 
       -0.13514, -0.17048, -0.13989, -0.14828, -0.16957, -0.16182, 
       -0.18837, -0.19360, -0.21499, -0.19963, -0.29463, -0.38204, 
       -0.29486, -0.30970, -0.38432, -0.37182, -0.36092, -0.34784, 
       -0.28937, -0.28408])

d_3 = np.array([-0.01602, -0.00996, -0.00006, 0.0008, -0.00503, 0.0097, 
       -0.00484, 0.00222, 0.00678, 0.01428, 0, 0.00347, 0.01248, 
       0.01754, 0.00412, 0.00958, 0.03053, 0.0083, 0.00978, 0.00568, 
       0.01146, 0.0294, 0.04706, 0.04902, 0.00116, 0.01237, 0.03473, 
       0.01845, 0.04707, -0.13444, -0.0111, 0.0101])

# remain constant
c_1 = 0.00034115 * np.ones_like(lamda)
c_2 = -0.0013425 * np.ones_like(lamda)
c_3 = 0.00095906 * np.ones_like(lamda)
c_4 = 0.00066229 * np.ones_like(lamda)
p_1 = 4.06054 * np.ones_like(lamda)
p_2 = 12.8802 * np.ones_like(lamda)
p_3 = -30.5858 * np.ones_like(lamda)
p_4 = 16.7498 * np.ones_like(lamda)


# celestial module
def transform_to_selenocentric(coords_j2000, et):
    # Get the transformation matrix from J2000 to lunar-centric frame (IAU_MOON)
    j2000_to_moon_matrix = spice.pxform('J2000', 'IAU_MOON', et)

    # Apply the transformation to get the position in lunar-centric coordinates
    coords_moon = spice.mxv(j2000_to_moon_matrix, coords_j2000)

    return coords_moon



def calculate_selenographic_coordinates(lunar_coords):
	# Assuming lunar_coords is a tuple (x, y, z) in a lunar-centric coordinate system
	
	x, y, z = lunar_coords
	
	# Calculate selenographic latitude (phi)
	# This is the angle between the equatorial plane and the line to the point
	latitude = np.arctan2(z, np.sqrt(x**2 + y**2)) 
	
	# Calculate selenographic longitude (lambda)
	# This is the angle in the equatorial plane from a reference direction
	longitude = np.arctan2(y, x)
	
	# Convert to degrees and adjust the range of longitude
	latitude_deg = np.degrees(latitude)
	longitude_deg = np.degrees(longitude) % 360
	
	if longitude_deg > 180:
		longitude_deg = longitude_deg - 360
	
	return latitude_deg, longitude_deg


def get_lunar_albedo(date, mpa, KERNELS_PATH = './KERNELS/meta.tm', ideal = False):
	
	
	# pay attention the mpa range -180 to 180 here
	spice.furnsh(KERNELS_PATH)
	
	UTC_string = date + 'T' + '00:00:00'
	et = spice.str2et(UTC_string)
	
	sun_pos_j2000, _ = spice.spkpos('Sun', et, 'J2000', 'LT+S', 'Moon')
	sel_sun = transform_to_selenocentric(sun_pos_j2000, et)
	sel_sun_cord = calculate_selenographic_coordinates(sel_sun)
# 		print(f' - selenographic coordinate for the Sun, latitude: {sel_sun_cord[0]}  longitude: {sel_sun_cord[1]}, {sel_sun_cord[1]-360}')
	sel_sun_lon_rad = np.deg2rad(sel_sun_cord[1])
	
	
	earth_pos_j2000, _ = spice.spkpos('Earth', et, 'J2000', 'LT+S', 'Moon')
	sel_earth = transform_to_selenocentric(earth_pos_j2000, et)
	sel_earth_cord = calculate_selenographic_coordinates(sel_earth)
# 		print(f' - selenographic coordinate for the Earth, latitude: {sel_earth_cord[0]}  longitude: {sel_earth_cord[1]}, {sel_earth_cord[1]-360}')
	sel_earth_lat = sel_earth_cord[0]
	sel_earth_lon = sel_earth_cord[1]
	
	
	abs_mpa = abs(mpa)
	abs_mpa_rad = abs(np.deg2rad(mpa))
	
	if ideal:
		print(' - get_lunar_albedo: use ideal geometry for reflectance calculation...')
		sel_sun_lon_rad = 0
		sel_earth_lat = 0
		sel_earth_lon = 0
		abs_mpa = abs(mpa)
		abs_mpa_rad = abs(np.deg2rad(mpa))

	# rolo Disk Reflectance model Eqn. 10
	ln_ak = a_0 + a_1*abs_mpa_rad**1 + a_2*abs_mpa_rad**2 + a_3*abs_mpa_rad**3  + \
			b_1*sel_sun_lon_rad**(2*1-1) + b_2*sel_sun_lon_rad**(2*2-1) + b_3*sel_sun_lon_rad**(2*3-1) + \
			c_1*sel_earth_lat + c_2*sel_earth_lon + c_3 * sel_sun_lon_rad * sel_earth_lat + c_4 * sel_sun_lon_rad * sel_earth_lon + \
			d_1*np.exp(-abs_mpa/p_1) + d_2*np.exp(-abs_mpa/p_2) + d_3*np.cos((abs_mpa - p_3)/p_4)
	
	ak = np.exp(ln_ak)

	return lamda, ak



if __name__ == '__main__':

	from pylib.convert_time_coordinate import *
	import spiceypy as spice
	from celestial import *
	import matplotlib.pyplot as plt
	
	
	
# 	gregDayBeg = '20240101'
# 	gregDayEnd = '20241231'
# 	dateSeries = get_date_series(gregDayBeg, gregDayEnd, outtype = 'greg')
# 	lat, lon, alt = 60.43567, -69.96872, 0 	
# 	lamda_idx = 30
# 	
# 	hour=0
# 	minute=0
# 	mpas = []
# 	aks = []
# 	
# 	for date in dateSeries:
# 		year, month, day = date.split('-')
# 		year = int(float(year))
# 		month = int(float(month))
# 		day = int(float(day))		
# 		zenith_sun, zenith_moon, moon_phase_angle, moon_illumination_fraction, utc_time = compute_zenith_and_moon_phase(lat, lon, year, month, day, hour, minute)
# # 		print(date, 'mpa', moon_phase_angle, 'mif', moon_illumination_fraction)
# 		
# 		abs_mpa = abs(moon_phase_angle - 180)
# 		abs_mpa_rad = abs(np.deg2rad(moon_phase_angle - 180) )
# 		
# 		ak = get_lunar_albedo(date)
# 		
# 		mpas.append( moon_phase_angle - 180 )
# 		aks.append(ak[lamda_idx])
# 		
# 
# 
# 	fig = plt.figure(figsize = (6,4))
# 	ax1   = fig.add_axes([0.2,0.2,0.6,0.6])
# 	ax1.plot(mpas, aks, '.')
# 	ax1.set_xlabel('Phase Angle (degree)')
# 	ax1.set_ylabel('Disk Reflectance')
# 	ax1.set_xlim(-100, 100)
# 	ax1.set_ylim(0.01, np.nanmax(aks)*1.2)
# 	ax1.set_yscale('log')
# 	ax1.set_title(f'Wavelength {lamda[lamda_idx]} nm')
# 	plt.savefig('./FIG/ROLO_Reflectance_'+ str(int(lamda[lamda_idx])) + 'nm.png', dpi = 300)


	#------
	dateSeries = ['2024-02-19', '2024-02-24', '2024-02-29']
	lat, lon, alt = 1.0, 0.0, 0 	
	lamda_idx = 30

	hour=0
	minute=0
	mpas = []
	aks = []
	mifs = []
	for date in dateSeries:
		year, month, day = date.split('-')
		year = int(float(year))
		month = int(float(month))
		day = int(float(day))		
		zenith_sun, zenith_moon, moon_phase_angle, moon_illumination_fraction, utc_time = compute_zenith_and_moon_phase(lat, lon, year, month, day, hour, minute)

		wl, ak = get_lunar_albedo(date, moon_phase_angle-180)
		
		mpas.append( moon_phase_angle - 180 )
		mifs.append(moon_illumination_fraction)
		aks.append(ak)
				
# 	print(abs_mpa, abs_mpa_rad)

	fig = plt.figure(figsize = (6,4))
	ax1   = fig.add_axes([0.2,0.2,0.6,0.6])
	
	for i, mif in enumerate(mifs):
	
		ax1.plot(lamda, aks[i], label = f'moonphase {mif*100:.2f}%')
	ax1.set_xlabel('Wavelength (nm)')
	ax1.set_ylabel('Disk Reflectance')
	ax1.set_ylim(0.01, np.nanmax(aks)*1.2)
	ax1.legend(frameon = False)
	ax1.set_yscale('log')
	plt.savefig('./FIG/ROLO_spectrum_Reflectance.png', dpi = 300)




	


