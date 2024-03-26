import spiceypy as spice
import numpy as np
from datetime import datetime


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
	
	return x/1000, y/1000, z/1000

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

def transform_to_selenocentric(coords_j2000, et):
    # Get the transformation matrix from J2000 to lunar-centric frame (IAU_MOON)
    j2000_to_moon_matrix = spice.pxform('J2000', 'IAU_MOON', et)

    # Apply the transformation to get the position in lunar-centric coordinates
    coords_moon = spice.mxv(j2000_to_moon_matrix, coords_j2000)

    return coords_moon


spice.furnsh('./KERNELS/meta.tm')



# Define Earth coordinates in WGS84
lat, lon, alt = 18.43567, -69.96872, 0
lat, lon, alt = 18.43567, -169.96872, 0
UTC_string = '2024-01-14T00:00:00'

print('UTC ', UTC_string)
# Get current time in Ephemeris Time
et = spice.str2et(UTC_string)
print(f" - Ephemeris Time (et) for the {UTC_string}:", et)


et_reference = spice.str2et('2000-01-01T12:00:00')


# Convert geographic coordinates to ECEF
ecef_coords = wgs84_to_ecef(lat, lon, alt)
ecef_coords_reference = wgs84_to_ecef(0, 0, -6378137.0)
print('ECEF:', ecef_coords)
print('ECEF reference:', ecef_coords_reference)


eci_coords = ecef_to_eci(ecef_coords, et)
eci_coords_reference = ecef_to_eci(ecef_coords_reference, et)
print('ECI:', eci_coords)
print('ECI reference:', eci_coords_reference)

earth_to_moon, _ = spice.spkpos('Earth', et, 'J2000', 'LT+S', 'Moon')

print('earth_to_moon', earth_to_moon)
eci_coords = eci_coords
print('ECI:')

pos_coordinate = earth_to_moon + (eci_coords - eci_coords_reference)


# Get the transformation matrix from J2000 to lunar-centric frame (IAU_MOON)
transformation_matrix = spice.pxform('J2000', 'IAU_MOON', et)
lunar_centric_coords = spice.mxv(transformation_matrix, pos_coordinate)
# Calculate selenographic coordinates
sel_lat, sel_lon = calculate_selenographic_coordinates(lunar_centric_coords)
print(f'Selenographic Latitude: {sel_lat}, Longitude: {sel_lon}, {sel_lon-360}')

