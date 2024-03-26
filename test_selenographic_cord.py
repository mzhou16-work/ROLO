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
	
	return x, y, z


# Rotate ECEF coordinates to get ECI coordinates
def ecef_to_eci_naivi(ecef, gst_degrees):
    theta = np.radians(gst_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, ecef)

# def ecef_to_eci(ecef_coords, et):
# 	# Conversion from ECEF frame to Earth-Centered Inertial (ECI) frame
# 	# Ensure the necessary SPICE kernels are loaded
# 	# spice.furnsh('path_to_kernel/meta.tm')
# 	
# 	# Get the rotation matrix from ECEF (IAU_EARTH) to ECI (J2000)
# 	rotation_matrix = spice.pxform('IAU_EARTH', 'J2000', et)
# 	
# 	print(rotation_matrix)
# 	
# 	# Convert ECEF coordinates to a NumPy array
# 	ecef_coords_array = np.array(ecef_coords)
# 	
# 	# Apply the rotation matrix to the ECEF coordinates
# 	eci_coords = np.dot(rotation_matrix, ecef_coords_array)
# 	
# 	return eci_coords
    
def ecef_to_eci(ecef_pos, ecef_vel, et):
	# Ensure the necessary SPICE kernels are loaded
	# spice.furnsh('path_to_kernel/meta.tm')
	
	# Get the rotation matrix from ECEF (IAU_EARTH) to ECI (J2000)
	# retrieves the rotation matrix that transforms coordinates from the 
	# ECEF frame (referred to as 'IAU_EARTH' in SPICE) to the ECI frame #
	# aligned with the J2000 epoch (referred to as 'J2000')
	rotation_matrix = spice.pxform('IAU_EARTH', 'J2000', et)
	
	print('rotation_matrix')
	print(rotation_matrix)
	
	# Convert ECEF position and velocity to NumPy arrays
	ecef_pos_array = np.array(ecef_pos)
	ecef_vel_array = np.array(ecef_vel)
	
	# Apply the rotation matrix to the ECEF position
	eci_pos = np.dot(rotation_matrix, ecef_pos_array)
	
	# Earth's angular velocity (rad/s) - approximate value
	earth_angular_velocity = np.array([0, 0, 7.2921159e-5])
	
	# Compute the derivative of the rotation matrix
	d_rotation_matrix = spice.sxform('IAU_EARTH', 'J2000', et)
	
	# Apply the transformation to velocity
	eci_vel = np.dot(rotation_matrix, ecef_vel_array) + np.cross(earth_angular_velocity, eci_pos)
	
	return eci_pos, eci_vel


def eci_to_selenocentric(eci_coords, et):
	# Load necessary SPICE kernels
	# Assume the kernels are already loaded in the script

	# Get the transformation matrix from ECI (J2000) to the Moon's principal axes
	# at the given time
	eci_to_moon_matrix = spice.pxform('J2000', 'IAU_MOON', et)
	
	# Transform the ECI coordinates to the Moon's principal axes
	selenocentric_coords = spice.mxv(eci_to_moon_matrix, eci_coords)
	
	return selenocentric_coords


def j2000_to_selenocentric(sun_pos_j2000, et):
    # Get the transformation matrix from J2000 to lunar-centric frame (IAU_MOON)
    j2000_to_moon_matrix = spice.pxform('J2000', 'IAU_MOON', et)

    # Apply the transformation to get the position in lunar-centric coordinates
    sun_pos_moon = np.dot(j2000_to_moon_matrix, sun_pos_j2000)

    return sun_pos_moon
    

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


def calculate_zenith_azimuth_angles(moon_pos, observer_pos):
    # Calculate the vector from the observer to the Moon
    moon_vector = np.array(moon_pos) - np.array(observer_pos)

    # Normalize the moon_vector
    moon_vector /= np.linalg.norm(moon_vector)

    # Zenith vector (straight up from the observer)
    zenith_vector = np.array(observer_pos)
    zenith_vector /= np.linalg.norm(zenith_vector)

    # Calculate Zenith Angle
    zenith_angle = np.arccos(np.dot(moon_vector, zenith_vector))

    # Calculate azimuth
    # Project moon_vector onto the Earth's surface plane and calculate angle from North
    north_vector = np.array([0, 1, 0])  # Assuming ECI frame where Y-axis points towards North
    east_vector = np.cross(zenith_vector, north_vector)
    moon_proj = moon_vector - np.dot(moon_vector, zenith_vector) * zenith_vector
    moon_proj /= np.linalg.norm(moon_proj)

    azimuth_angle = np.arctan2(np.dot(moon_proj, east_vector), np.dot(moon_proj, north_vector))

    return np.degrees(zenith_angle), (np.degrees(azimuth_angle) + 360) % 360


def calculate_ecef_velocity(lat, lon, alt):
    # Earth's rotation rate (radians per second)
    omega = 7.2921159e-5  

    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Radius of the Earth at the given latitude (approximation)
    R = 6378137  # Radius of Earth in meters at the equator
    r = R * np.cos(lat_rad)

    # Velocity due to Earth's rotation
    vx = -r * omega * np.sin(lon_rad)
    vy = r * omega * np.cos(lon_rad)
    vz = 0  # No vertical component in this simple model

    return [vx, vy, vz]

print('======================')
# Step 1: Load kernels
spice.furnsh('./KERNELS/meta.tm')

# Define Earth coordinates in WGS84
lat, lon, alt = 18.43567,-69.96872,0  # Example: New York City
UTC_string = '2019-01-01T12:00:00'

print('UTC ', UTC_string)
# Get current time in Ephemeris Time
et = spice.str2et('2019-01-01T12:00:00')
print(" - Ephemeris Time (et) for the given date and time:", et)


# Convert Earth coordinates to ECEF
ecef_coords = wgs84_to_ecef(lat, lon, alt)
ecef_vel = calculate_ecef_velocity(lat, lon, alt)
print(" - Earth-Centered, Earth-Fixed frame coordinate for the given point:", ecef_coords, ecef_vel)

# Transform ECEF to ECI
# eci_coords = ecef_to_eci(ecef_coords, et)

eci_coords, eci_vel = ecef_to_eci(ecef_coords, ecef_vel, et)
print(" - Earth-Centered Inertial frame for the given point [T]:", eci_coords)
# print(" - Earth-Centered Inertial frame for the given point:", eci_vel)

# Get position of Moon and Sun relative to Earth
moon_pos, _ = spice.spkpos('Moon', et, 'J2000', 'CN+S', 'Earth')
print('*', moon_pos)


from datetime import datetime

def datetime_to_julian_date(dt):
    # Convert a datetime object to Julian date
    # Reference: https://en.wikipedia.org/wiki/Julian_day
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (dt.hour - 12) / 24 + dt.minute / 1440 + dt.second / 86400
    return jd

def lst_from_utc(utc_time, longitude_deg):
    # Convert UTC to LST. This is a simplified approximation.
    # For high precision, use an astronomical library to account for the Earth's precession and nutation.
    # utc_time is a datetime object
    
    jd = datetime_to_julian_date(utc_time)
    jd0 = np.floor(jd + 0.5) - 0.5  # Julian date at preceding midnight
    h = (jd - jd0) * 24.0  # Hours since midnight
    d = jd - 2451545.0  # Days since J2000
    d0 = jd0 - 2451545.0
    t = d / 36525.0  # Julian centuries since J2000

    # Greenwich Mean Sidereal Time (GMST) at 0h UT at Greenwich
    gmst = 6.697374558 + 0.06570982441908 * d0 + 1.00273790935 * h + 0.000026 * t**2

    # Convert longitude to hours
    long_hours = longitude_deg / 15.0

    # Local Sidereal Time (LST)
    lst = (gmst + long_hours) % 24.0
    return lst


def eci_to_topocentric_az_zenith(moon_pos_relative, observer_lat, observer_lon, utc_time):
    # Convert observer's coordinates to radians
    observer_lat_rad = np.radians(observer_lat)
    observer_lon_rad = np.radians(observer_lon)

    # Calculate Local Sidereal Time
    lst_hours = lst_from_utc(utc_time, observer_lon)
    lst_rad = np.radians(lst_hours * 15.0)  # Convert LST to radians

    # First rotation matrix (for LST)
    R1 = np.array([
        [np.cos(lst_rad), np.sin(lst_rad), 0],
        [-np.sin(lst_rad), np.cos(lst_rad), 0],
        [0, 0, 1]
    ])

    # Second rotation matrix (for latitude)
    R2 = np.array([
        [-np.sin(observer_lat_rad), 0, np.cos(observer_lat_rad)],
        [0, 1, 0],
        [np.cos(observer_lat_rad), 0, np.sin(observer_lat_rad)]
    ])

    # Apply rotations
    topocentric_vector = R2 @ R1 @ moon_pos_relative

    # Convert to spherical coordinates
    x, y, z = topocentric_vector
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)  # Azimuth
    zenith = np.arccos(z / r)  # Zenith angle

    # Convert to degrees
    azimuth_deg = np.degrees(azimuth)
    zenith_deg = np.degrees(zenith)

    # Adjust azimuth range from [-180, 180] to [0, 360]
    if azimuth_deg < 0:
        azimuth_deg += 360

    return azimuth_deg, zenith_deg

utc_time = datetime.strptime(UTC_string.replace('T', ' '), '%Y-%m-%d %H:%M:%S')

lst = lst_from_utc(utc_time, lon)


print('lst: ', lst)
moon_pos_relative = np.array(moon_pos) - ecef_coords
azimuth_deg, zenith_deg = eci_to_topocentric_az_zenith(moon_pos_relative, lat, lon, utc_time)

print(azimuth_deg, zenith_deg)

# # Transform Moon's position from ECI to topocentric horizon coordinates
# moon_pos_relative = np.array(moon_pos) - ecef_coords
# topocentric_moon_pos = np.dot(R, moon_pos_relative)
# 
# # Calculate zenith and azimuth angles
# zenith_angle, azimuth_angle = calculate_zenith_azimuth(topocentric_moon_pos)
# 
# print("Moon Zenith Angle:", zenith_angle)
# print("Moon Azimuth Angle:", azimuth_angle)





print('-----------')
from skyfield.api import Topos, load

# def topos_to_ecef(topos, altitude_m=0):
#     # Constants
#     a = 6378137.0  # Earth's equatorial radius in meters
#     f = 1 / 298.257223563  # Flattening
#     b = a * (1 - f)  # Polar radius
# 
#     # Convert latitude and longitude to radians
#     lat_rad = np.radians(topos.latitude.degrees)
#     lon_rad = np.radians(topos.longitude.degrees)
# 
#     # Calculate ECEF coordinates
#     cos_lat = np.cos(lat_rad)
#     sin_lat = np.sin(lat_rad)
#     cos_lon = np.cos(lon_rad)
#     sin_lon = np.sin(lon_rad)
# 
#     N = a / np.sqrt(1 - f * (2 - f) * sin_lat**2)
#     X = (N + altitude_m) * cos_lat * cos_lon
#     Y = (N + altitude_m) * cos_lat * sin_lon
#     Z = ((b**2 / a**2) * N + altitude_m) * sin_lat
# 
#     return np.array([X, Y, Z])


# Load ephemeris data
ts = load.timescale()
eph = load('de440.bsp')


# Define the observation time
t = ts.utc(2024, 2, 11, 12, 0, 0)
print('Time info')
print(t, t.gast, t.gast*360, t.M)

location = Topos(latitude_degrees=lat, longitude_degrees=lon)
print(location)

# ecef_coordinates = topos_to_ecef(location)
# print("ECEF Coordinates (meters):", ecef_coordinates)


#-----------------------------------------------------------------------
# Get Earth's position from the solar system barycenter
earth = eph['earth']
earth_position = earth.at(t).position.km


print(f' - earth position: {earth_position}')

# Get the position of the Moon relative to the Earth, in the J2000 frame
earth = eph['earth']
moon = eph['moon']
moon_position = earth.at(t).observe(moon).apparent()


observer_position_ssb = (location + earth).at(t)
print("Observer location at Solar System Barycenter system [t]:", observer_position_ssb.position.km)

# Get Earth's center position in the J2000 frame relative to the SSB
earth_position_j2000 = earth.at(t).observe(eph['earth barycenter']).apparent().position.km
print("Earth's center position in the J2000 frame relative to the SSB [t]:", earth_position_j2000)

# Convert to the ECI J2000 frame
observer_position_eci_j2000 = observer_position_ssb.position.km - earth_position_j2000
print("Observer location in ECI J2000 (km)  [t]:", observer_position_eci_j2000)


# Print the Moon's position in the ECI J2000 frame
print("Moon's Position in ECI J2000 Frame (km) [t]:", moon_position.position.km)
print('--')
moon_position = (location + earth).at(t).observe(moon).apparent().altaz()
print("Moon Zenith, Azimuth:", 90 - moon_position[0].degrees, moon_position[1].degrees)


'''

astrometric = (location + eph['earth']).at(t).observe(moon).apparent()


alt, az, distance = astrometric.altaz()




# Print Moon's Altitude and Azimuth
print("Moon Zenith:", 90-alt.degrees)
print("Moon Azimuth:", az.degrees)

'''


# def topocentric_horizon_transform(observer_ecef, lat_rad, lon_rad):
#     # Define rotation matrix from ECEF to topocentric horizon coordinates
#     sin_lat = np.sin(lat_rad)
#     cos_lat = np.cos(lat_rad)
#     sin_lon = np.sin(lon_rad)
#     cos_lon = np.cos(lon_rad)
# 
#     R = np.array([[-sin_lon, cos_lon, 0],
#                   [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
#                   [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]])
# 
#     return R
# 
# def calculate_zenith_azimuth(topocentric_moon_pos):
#     x, y, z = topocentric_moon_pos
# 
#     # Calculate azimuth
#     azimuth = np.arctan2(y, x)
#     azimuth_degrees = np.degrees(azimuth) % 360
# 
#     # Calculate zenith angle
#     horizontal_distance = np.sqrt(x**2 + y**2)
#     zenith_angle = np.arctan2(horizontal_distance, z)
#     zenith_angle_degrees = 90 - np.degrees(zenith_angle)
# 
#     return zenith_angle_degrees, azimuth_degrees
# 
# # Convert to radians
# lat_rad = np.radians(lat)
# lon_rad = np.radians(lon)
# 
# # Convert WGS84 to ECEF (X, Y, Z)
# radius_earth = 6378137.0  # Earth's radius in meters
# f = 1 / 298.257223563  # Flattening factor
# cos_lat = np.cos(lat_rad)
# sin_lat = np.sin(lat_rad)
# N = radius_earth / np.sqrt(1 - f * (2 - f) * sin_lat**2)
# x = (N + alt) * cos_lat * np.cos(lon_rad)
# y = (N + alt) * cos_lat * np.sin(lon_rad)
# z = ((1 - f)**2 * N + alt) * sin_lat
# 
# # Convert ECEF to ECI if necessary (depends on your application)
# 
# # Observer's ECEF position
# observer_ecef = np.array([x, y, z])
# 
# # Compute Moon's position relative to the Earth's center in ECI (J2000 frame)
# moon_pos, _ = spice.spkpos('Moon', et, 'J2000', 'NONE', 'Earth')
# 
# # Compute topocentric transformation matrix
# R = topocentric_horizon_transform(observer_ecef, lat_rad, lon_rad)
# 
# # Transform Moon's position from ECI to topocentric horizon coordinates
# moon_pos_relative = np.array(moon_pos) - observer_ecef
# topocentric_moon_pos = np.dot(R, moon_pos_relative)
# 
# # Calculate zenith and azimuth angles
# zenith_angle, azimuth_angle = calculate_zenith_azimuth(topocentric_moon_pos)
# 
# print("Moon Zenith Angle:", zenith_angle)
# print("Moon Azimuth Angle:", azimuth_angle)









'''

sun_pos_j2000, _ = spice.spkpos('Sun', et, 'J2000', 'NONE', 'Moon')

#  J2000 coordinates to selenocentric
sel_points = transform_to_selenocentric(eci_coords, et)
print(sel_points)

sel_sun = transform_to_selenocentric(sun_pos_j2000, et)


# Calculate selenographic latitude and longitude
sel_lat_point, sel_lon_point = calculate_selenographic_coordinates(sel_points)
print(' - Selenographic coordinates of given point:', sel_lat_point, sel_lon_point)
# Calculate selenographic coordinates of the Sun
sel_lat_sun, sel_lon_sun = calculate_selenographic_coordinates(sel_sun)
print(' - Selenographic coordinates of the Sun:', sel_lat_sun, sel_lon_sun)


# observer_pos = [x, y, z] in ECI coordinates
zenith_angle, azimuth_angle = calculate_zenith_azimuth_angles(moon_pos, eci_coords)


print("Zenith", zenith_angle)
print("azimuth", azimuth_angle)

'''

# zenith_angle, azimuth_angle = calculate_zenith_azimuth_angles(moon_pos, eci_coords)
# 
# print("Zenith:", zenith_angle)
# print("Azimuth:", azimuth_angle)



'''
from skyfield.api import Topos, load

# Load ephemeris data
ts = load.timescale()
eph = load('de440.bsp')


# Define the observation time
t = ts.utc(2019, 1, 1, 12, 0, 0)

# Get the position of the Moon relative to the Earth
earth = eph['earth']
moon = eph['moon']

# Compute the position in the ICRF (which is similar to ECI) frame
astrometric = earth.at(t).observe(moon)
moon_eci = astrometric.position.km
# position = astrometric.position.au  # Position in Astronomical Units
# 
# # Convert position to kilometers (1 AU = 149597870.7 km)
# position_km = position * 149597870.7

# Extracting the X, Y, Z components
x, y, z = moon_eci

print("Moon position in ECI (X, Y, Z) in kilometers:", x, y, z)

location = Topos(latitude_degrees=lat, longitude_degrees=lon)
observer_eci = location.at(t).position.km
print(observer_eci)

relative_position = eci_coords - observer_eci

print(relative_position)

    
# Convert relative position to topocentric horizontal coordinates
# Rotate the relative position vector into the topocentric frame
lat, lon = location.latitude.radians, location.longitude.radians
sin_lat, cos_lat = np.sin(lat), np.cos(lat)
sin_lon, cos_lon = np.sin(lon), np.cos(lon)

# Rotation matrix
rotation_matrix = np.array([
	[-sin_lon, cos_lon, 0],
	[-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
	[cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
])

topocentric_position = np.dot(rotation_matrix, relative_position)

print(topocentric_position)

def calculate_zenith_azimuth_angles(topocentric_position):
    # Extract topocentric coordinates
    x, y, z = topocentric_position

    # Calculate azimuth
    azimuth = np.arctan2(y, x)
    azimuth_degrees = np.degrees(azimuth) % 360

    # Calculate zenith angle
    horizontal_distance = np.sqrt(x**2 + y**2)
    zenith_angle = np.arctan2(horizontal_distance, z)
    print( np.degrees(zenith_angle))
    zenith_angle_degrees = 90 - np.degrees(zenith_angle)

    return zenith_angle_degrees, azimuth_degrees
    
zenith_angle_degrees, azimuth_degrees = calculate_zenith_azimuth_angles(topocentric_position)
print(zenith_angle_degrees, azimuth_degrees)


'''

# from skyfield.api import Topos, load
# 
# # Load ephemeris data
# ts = load.timescale()
# eph = load('de440.bsp')
# 
# 
# # Define the observation time
# t = ts.utc(2019, 1, 1, 12, 0, 0)
# location = Topos(latitude_degrees=lat, longitude_degrees=lon)
# #-----------------------------------------------------------------------
# # Get the position of the Moon relative to the observer's location
# moon = eph['moon']
# astrometric = (location + eph['earth']).at(t).observe(moon).apparent()
# alt, az, distance = astrometric.altaz()
# 
# # Print Moon's Altitude and Azimuth
# print("Moon Zenith:", 90-alt.degrees)
# print("Moon Azimuth:", az.degrees)

'''
print(' II ')
from skyfield.api import Topos, load

# Define the observation time
t = ts.utc(2019, 1, 1, 12, 0, 0)

# Get the position of the Moon relative to the Earth
earth  = eph['earth']
sun    = eph['sun']
moon   = eph['moon']


Haifa   = earth + location

print(Haifa)

print(dir(Haifa))

salt, saz, sd = Haifa.at(t).observe(sun).apparent().altaz()
malt, maz, md = Haifa.at(t).observe(moon).apparent().altaz()
print(salt.degrees, saz.degrees, malt.degrees, maz.degrees)

saltr, sazr, maltr, mazr = [x.radians for x in (salt, saz, malt, maz)]

print(saltr, sazr, maltr, mazr)
'''

# # Define the observer's location
# location = Topos(latitude_degrees=18.43567, longitude_degrees=-69.96872, elevation_m=0)
# 
# # Load ephemeris data
# ts = load.timescale()
# eph = load('de440.bsp')
# # Define the observation time
# # t = ts.utc(2019, 1, 21, 4, 41, 44)
# t = ts.utc(2019, 1, 1, 12, 0, 0)
# print('-------')
# print( (t.tt - 2451545.0)* 86400) 
# # Get the position of the Moon relative to the observer's location
# moon = eph['moon']
# astrometric = (location + eph['earth']).at(t).observe(moon).apparent()
# alt, az, distance = astrometric.altaz()
# 
# # Print Moon's Altitude and Azimuth
# print("Moon Altitude:", alt)
# print("Moon Azimuth:", az)
