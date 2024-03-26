import spiceypy as spice
import numpy as np
from datetime import datetime

# https://clearskytonight.com/projects/astronomycalculator/modification/selenographic_moon.html

def wgs84_to_ecef(lat, lon, alt):
	
	# Earth-Centered, Earth-Fixed (ECEF) frame

	# WGS84 ellipsoid constants
	a = 6378137.0  # Semi-major axis
	e = 8.1819190842622e-2  # First eccentricity
	
	# Convert latitude, longitude, and altitude to radians
	lat_rad = np.radians(lat)
	lon_rad = np.radians(lon)
	
	# Calculate prime vertical radius of curvature
	N = a / np.sqrt(1 - e**2 * np.sin(lat_rad)**2)
	
	# Calculate ECEF coordinates
	x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
	y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
	z = ((1 - e**2) * N + alt) * np.sin(lat_rad)
	
	return x, y, z

def ecef_to_eci(ecef_coords, et):
	# Conversion from ECEF frame to Earth-Centered Inertial (ECI) frame
	# Ensure the necessary SPICE kernels are loaded
	# spice.furnsh('path_to_kernel/meta.tm')
	
	# Get the rotation matrix from ECEF (IAU_EARTH) to ECI (J2000)
	rotation_matrix = spice.pxform('IAU_EARTH', 'J2000', et)

	# Convert ECEF coordinates to a NumPy array
	ecef_coords_array = np.array(ecef_coords)
	
	# Apply the rotation matrix to the ECEF coordinates
	eci_coords = np.dot(rotation_matrix, ecef_coords_array)
	
	return eci_coords

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

    return latitude_deg, longitude_deg


'''
KPL/MK

\begindata

   PATH_VALUES  = ('./KERNELS/')
   PATH_SYMBOLS = ('KERNELS')

   KERNELS_TO_LOAD = ( '$KERNELS/de440.bsp',
                       '$KERNELS/naif0012.tls',
                       '$KERNELS/pck00011.tpc')

\begintext

'''
   
spice.furnsh('./KERNELS/meta.tm')

# Define Earth coordinates in WGS84
lat, lon, alt = 18.43567, -69.96872, 0
lat, lon, alt = 60.43567, -69.96872, 0 
UTC_string = '2024-01-14T00:00:00'

print('UTC ', UTC_string)
# Get current time in Ephemeris Time
et = spice.str2et(UTC_string)
print(f" - Ephemeris Time (et) for the {UTC_string}:", et)

# Convert Earth coordinates to ECEF
ecef_coords = wgs84_to_ecef(lat, lon, alt)


# Transform ECEF to ECI
eci_coords = ecef_to_eci(ecef_coords, et)
print(" - Earth-Centered Inertial frame for the given point [T]:", eci_coords)

sun_pos_j2000, _ = spice.spkpos('Sun', et, 'J2000', 'LT+S', 'Moon')


#  J2000 to selenocentric
sel_points = transform_to_selenocentric(eci_coords, et)
sel_sun = transform_to_selenocentric(sun_pos_j2000, et)

sel_points_cord = calculate_selenographic_coordinates(sel_points)
sel_sun_cord = calculate_selenographic_coordinates(sel_sun)

print(f' - selenographic coordinate for given point, latitude: {sel_points_cord[0]}  longitude: {sel_points_cord[1]}')
print(f' - selenographic coordinate for the Sun, latitude: {sel_sun_cord[0]}  longitude: {sel_sun_cord[1]}, {sel_sun_cord[1]-360}')


earth_pos_j2000, _ = spice.spkpos('Earth', et, 'J2000', 'LT+S', 'Moon')
print('earth_pos_j2000', earth_pos_j2000, _)
sel_earth = transform_to_selenocentric(earth_pos_j2000, et)
sel_earth_cord = calculate_selenographic_coordinates(sel_earth)
print(f' - selenographic coordinate for the Earth, latitude: {sel_earth_cord[0]}  longitude: {sel_earth_cord[1]}, {sel_earth_cord[1]-360}')





