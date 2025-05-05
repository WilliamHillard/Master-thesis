import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Gravitational Constant (m^3 kg^-1 s^-2)
G = 6.67430e-11

# Masses (kg)
M_sun = 1.989e30
mu_sun = G * M_sun
M_jupiter = 1.898e27

# Jupiter's Orbital Elements
a_jupiter = 7.7834276e11  # Semi-major axis (m)
e_jupiter = 0.0484  # Eccentricity
mean_jupiter = np.radians(0)  # True anomaly at epoch (radians)
i_jupiter = np.radians(0)  # Inclination (radians) 1.304
omega_jupiter = np.radians(0)  # Argument of periapsis (radians) 274.3
Omega_jupiter = np.radians(0)  # Longitude of ascending node (radians) 100.4

# Earth's Orbital Elements
a_third = 1.496e11
e_third = 0.0167
mean_third = np.radians(0)
i_third = np.radians(0)
omega_third = np.radians(0)
Omega_third = np.radians(0)


def keplers_equation(E, M, e):
    return E - e * np.sin(E) - M


def solve_keplers_equation(M, e):
    E_initial_guess = M if e < 0.8 else np.pi
    E_solution, = fsolve(keplers_equation, E_initial_guess, args=(M, e))
    return E_solution


def orbital_elements_to_polar(a, e, M, mu):
    """
    Convert orbital elements to polar coordinates (r, theta) and velocities (vr, vθ) in SI units.

    Parameters:
        a  - Semi-major axis (meters)
        e  - Eccentricity (unitless)
        M  - Mean anomaly (radians)
        mu - Gravitational parameter (m^3/s^2)

    Returns:
        r  - Radius (meters)
        theta - True anomaly (radians)
        v_r - Radial velocity (m/s)
        v_theta - Tangential velocity (m/s)
    """
    # Solve Kepler's Equation to get the eccentric anomaly E
    E = solve_keplers_equation(M, e)

    # Compute true anomaly θ from eccentric anomaly E
    theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    # Compute the radius r
    r = a * (1 - e * np.cos(E))

    # Compute velocity components in polar coordinates
    h = np.sqrt(mu * a * (1 - e ** 2))  # Specific angular momentum
    v_r = (mu / h) * e * np.sin(theta)  # Radial velocity (m/s)
    v_theta = (mu / h) * (1 + e * np.cos(theta))  # Tangential velocity (m/s)

    return r, theta, v_r, v_theta


def orbital_elements_to_cartesian(a, e, M, mu):
    """
    Convert orbital elements to Cartesian coordinates (x, y) and velocities (vx, vy) in SI units.

    Parameters:
        a  - Semi-major axis (meters)
        e  - Eccentricity (unitless)
        M  - Mean anomaly (radians)
        mu - Gravitational parameter (m^3/s^2)

    Returns:
        x  - x-position (meters)
        y  - y-position (meters)
        vx - x-velocity (m/s)
        vy - y-velocity (m/s)
    """
    r, theta, v_r, v_theta = orbital_elements_to_polar(a, e, M, mu)

    # Convert polar to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Convert polar velocities to Cartesian velocities
    vx = v_r * np.cos(theta) - v_theta * np.sin(theta)
    vy = v_r * np.sin(theta) + v_theta * np.cos(theta)

    return np.array([x, y]), np.array([vx, vy])


def orbital_energy(r, v, mu):
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


# Function to Compute Orbital Angular Momentum
def orbital_angular_momentum(r, v):
    r_3d = np.array([r[0], r[1], 0])
    v_3d = np.array([v[0], v[1], 0])
    return np.linalg.norm(np.cross(r_3d, v_3d))


# Function to Compute Eccentricity
def compute_eccentricity(r, v, mu):
    h = orbital_angular_momentum(r, v)
    epsilon = orbital_energy(r, v, mu)
    e = np.sqrt(1 + (2 * epsilon * h ** 2) / (mu ** 2))
    return e


# Function to Compute Acceleration
def compute_acceleration(r1, r2, m2):
    r_vec = r2 - r1
    r_mag = np.linalg.norm(r_vec)
    return G * m2 * r_vec / r_mag ** 3


# Convert Orbital Elements to Cartesian Coordinates
jupiter_position, jupiter_velocity = orbital_elements_to_cartesian(a_jupiter, e_jupiter, mean_jupiter, mu_sun)
third_position, third_velocity = orbital_elements_to_cartesian(a_third, e_third, mean_third, mu_sun)

# Leapfrog Integration Setup
time_steps = 4000
dt = 86400
t = 0
iterations = 40
time_array = []
ecc_jupiter = []
final_ecc = []
mass = []

r_j = jupiter_position
v_j = jupiter_velocity
# r_third = third_position
# v_third = third_velocity
r_third = np.array([7e11, 0], dtype=np.float64)
v_third = np.array([0, 10000], dtype=np.float64)
M_third = 8.7e27  # 9.002e27  5.97219e24

# Simulation loop
for x in range(iterations):
    # Reset variables
    r_j = jupiter_position.copy()
    v_j = jupiter_velocity.copy()
    r_third = np.array([7e11, 0], dtype=np.float64)
    v_third = np.array([0, 10000], dtype=np.float64)
    ecc_jupiter = []

    for step in range(time_steps):
        v_j += 0.5 * dt * (
                compute_acceleration(r_j, np.array([0, 0]), M_sun) + compute_acceleration(r_j, r_third, M_third))
        v_third += 0.5 * dt * (
                compute_acceleration(r_third, np.array([0, 0]), M_sun) + compute_acceleration(r_third, r_j,
                                                                                              M_jupiter))
        r_j += v_j * dt
        r_third += v_third * dt

        acc_j = compute_acceleration(r_j, np.array([0, 0]), M_sun) + compute_acceleration(r_j, r_third, M_third)
        acc_third = compute_acceleration(r_third, np.array([0, 0]), M_sun) + compute_acceleration(r_third, r_j,
                                                                                                  M_jupiter)
        v_j += 0.5 * dt * acc_j
        v_third += 0.5 * dt * acc_third

        e_jupiter = compute_eccentricity(r_j, v_j, mu_sun)
        ecc_jupiter.append(e_jupiter)

    final_ecc.append(ecc_jupiter[-1])
    mass.append(M_third)
    M_third += 1e26


plt.figure()
plt.scatter(mass, final_ecc)
plt.xlabel('Third body mass (kg)')
plt.ylabel('Final eccentricity')
plt.title('Final Jupiter eccentricity')
plt.grid()
plt.show()
