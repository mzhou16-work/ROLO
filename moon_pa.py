import numpy as np

def moon_pa1(jdate):
    # Assuming jplephem and matran functions are defined elsewhere

    # Transformation matrix from lunar mean equator and IAU node of J2000
    # to Earth mean equator and equinox of J2000 (EME2000)
    tmatrix1 = np.array([[0.998496505205088, -5.481540926807404e-2, 0.0],
                         [4.993572939853833e-2, 0.909610125238044, 0.412451018902688],
                         [-2.260867140418499e-2, -0.411830900942612, 0.910979778593430]])

    # Compute lunar libration angles (radians)
    icent = 0
    itarg = 15
    sv = jplephem(jdate, itarg, icent)

    phi = sv[0]
    theta = sv[1]
    psi = sv[2] % (2 * np.pi)

    # Compute lunar libration matrix
    tmatrix2 = matran(phi, 3, theta, 1, psi, 3, 0.0, 0)

    # Create moon_j2000 to lunar principal axes transformation matrix
    tmatrix = np.dot(tmatrix2, tmatrix1)

    return tmatrix

# Example usage
# jdate = ... (Julian date)
# tmatrix = moon_pa1(jdate)
