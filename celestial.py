from skyfield.api import load, Topos
from skyfield.nutationlib import iau2000b
import numpy as np
from pylib.datawritepy import *
from pylib.convert_time_coordinate import *
import sys
import spiceypy as spice


def compute_zenith_and_moon_phase(latitude, longitude, year, month, day, hour, minute):
	"""
	Compute solar and lunar zenith angles, the moon phase angle, and moon illumination fraction for a given location and time.

	Parameters:
	- latitude (float): Latitude of the location.
	- longitude (float): Longitude of the location.
	- year, month, day (int): Date components.
	- hour, minute (int): Time components.

	Returns:
	- zenith_sun (float): Solar zenith angle in degrees.
	- zenith_moon (float): Lunar zenith angle in degrees.
	- moon_phase_angle (float): Moon phase angle in degrees.
	- moon_illumination_fraction (float): Fraction of the moon illuminated.
	- utc_time (skyfield.timelib.Time): Computed UTC time.
	"""

	# Load planetary ephemeris
	planets = load('de421.bsp')
	earth, moon, sun = planets['earth'], planets['moon'], planets['sun']

	# Build the topos object for the given latitude and longitude
	observer = earth + Topos(latitude, longitude)

	# Convert local time to UTC
	ts = load.timescale()
	local_time = ts.utc(year, month, day, hour, minute)
	delta_t = longitude / 360.0 * 24.0  # time difference in hours
	utc_time = ts.utc(year, month, day, hour - delta_t, minute)

	# Calculate positions
	astrometric_sun = (sun - observer).at(utc_time)
	astrometric_moon = (moon - observer).at(utc_time)

	alt_sun, az_sun, d_sun = astrometric_sun.altaz()
	alt_moon, az_moon, d_moon = astrometric_moon.altaz()

	# Calculate zenith angles
	zenith_sun = 90.0 - alt_sun.degrees
	zenith_moon = 90.0 - alt_moon.degrees

	# Calculate moon phase angle
	e = earth.at(utc_time)
	_, slon, _ = e.observe(sun).ecliptic_latlon()
	_, mlon, _ = e.observe(moon).ecliptic_latlon()
	phase_angle = (mlon.degrees - slon.degrees) % 360
	# should be 180 - phase_angle
	
	# Calculate moon illumination fraction
	moon_illumination_fraction = 0.5 * (1 - np.cos(np.deg2rad(phase_angle)))
	
	return zenith_sun, zenith_moon, phase_angle, moon_illumination_fraction, utc_time

