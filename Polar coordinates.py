import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Gravitational Constant (m^3 kg^-1 s^-2)
G = 6.67430e-11

# Masses (kg)
M_sun = 1.989e30
mu_sun = G * M_sun
M_jupiter = 1.898e27
M_third = 5.972e24  # Earth

# Jupiter's Orbital Elements
a_jupiter = 7.7834276e11  # Semi-major axis (m)
e_jupiter = 0.0484  # Eccentricity
i_jupiter = np.radians(0)
omega_jupiter = np.radians(0)
Omega_jupiter = np.radians(0)
theta_jupiter = np.radians(0)

# Earth's Orbital Elements
a_third = 1.496e11
e_third = 0.0167
i_third = np.radians(0)
omega_third = np.radians(0)
Omega_third = np.radians(0)
theta_third = np.radians(0)


def keplers_equation(E, M, e):
    return E - e * np.sin(E) - M


def solve_keplers_equation(M, e):
    E_initial_guess = M  # if e < 0.8 else np.pi
    E_solution, = fsolve(keplers_equation, E_initial_guess, args=(M, e))
    return E_solution


def orbital_elements_to_polar(a, e, M, mu):
    """
    Convert orbital elements to polar coordinates (r, theta) and velocities (vr, vθ).

    Parameters:
        a  - Semi-major axis (AU or other consistent unit)
        e  - Eccentricity (unitless)
        M  - Mean anomaly (radians)
        mu - Gravitational parameter (GM, in consistent units)

    Returns:
        r  - Radius (distance from the star)
        theta - True anomaly (radians)
        vr - Radial velocity
        vθ - Tangential velocity
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

    return np.array([r, theta]), np.array([v_r, v_theta])


# Function to Compute Orbital Energy
def orbital_energy(r, v, mu):
    return 0.5 * (v[0] ** 2 + v[1] ** 2) - mu / r[0]


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


# Function to Compute Total Energy
def total_energy(r_j, v_j, r_t, v_t):
    KE = 0.5 * M_jupiter * (v_j[0] ** 2 + v_j[1] ** 2) + 0.5 * M_third * (v_t[0] ** 2 + v_t[1] ** 2)
    PE_sun = -G * M_sun * (M_jupiter / r_j[0] + M_third / r_t[0])
    r_diff = np.sqrt(r_j[0] ** 2 + r_t[0] ** 2 - 2 * r_j[0] * r_t[0] * np.cos(r_t[1] - r_j[1]))
    PE_interaction = -G * M_jupiter * M_third / r_diff
    return KE + PE_sun + PE_interaction


# Function to Compute Total Angular Momentum
def total_angular_momentum(r_j, v_j, r_t, v_t):
    r_j_3d = np.array([r_j[0], r_j[1], 0])
    v_j_3d = np.array([v_j[0], v_j[1], 0])
    r_t_3d = np.array([r_t[0], r_t[1], 0])
    v_t_3d = np.array([v_t[0], v_t[1], 0])

    mom_jupiter = M_jupiter * np.cross(r_j_3d, v_j_3d)
    mom_third = M_third * np.cross(r_t_3d, v_t_3d)
    return np.linalg.norm(mom_jupiter + mom_third)


# Function to Compute Acceleration in Spherical Coordinates
def compute_acceleration(r1, r2, m2):
    x1 = r1[0] * np.cos(r1[1])
    y1 = r1[0] * np.sin(r1[1])
    x2 = r2[0] * np.cos(r2[1])
    y2 = r2[0] * np.sin(r2[1])
    x_diff = x2 - x1
    y_diff = y2 - y1
    r_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)
    theta_diff = np.arctan2(y_diff, x_diff)
    unit_vector = np.array([x_diff, y_diff]) / r_diff
    return G * m2 * unit_vector / r_diff ** 2


def compute_polar_acceleration(r1, r2, v1, m2): # WRONG???????????????????????
    dx = r2[0] * np.cos(r2[1]) - r1[0] * np.cos(r1[1])  # x separation
    dy = r2[0] * np.sin(r2[1]) - r1[0] * np.sin(r1[1])  # y separation
    dr = np.sqrt(dx ** 2 + dy ** 2)  # Separation

    acc_r = G * m2 / dr ** 3
    acc_theta = - 2 * v1[0] * v1[1] / r1[0]
    return np.array([acc_r, acc_theta])


# Convert Orbital Elements to Spherical Coordinates
jupiter_position, jupiter_velocity = orbital_elements_to_polar(a_jupiter, e_jupiter, theta_jupiter, mu_sun)
third_position, third_velocity = orbital_elements_to_polar(a_third, e_third, theta_third, mu_sun)

# Leapfrog Integration Setup
time_steps = 10000
dt = 86400  # 1 day
t = 0
time_array = []

r_pos_jupiter = []
theta_pos_jupiter = []
r_pos_third = []
theta_pos_third = []

total_energy_array = []
total_moment_array = []
ecc_jupiter = []
ecc_third = []
ener_jupiter = []
ener_third = []
moment_jupiter = []
moment_third = []
r_jupiter, theta_jupiter = jupiter_position
v_r_jupiter, v_theta_jupiter = jupiter_velocity
r_third, theta_third = third_position
v_r_third, v_theta_third = third_velocity

# Simulation loop using Leapfrog integration
for step in range(time_steps):
    jupiter_velocity += 0.5 * dt * compute_polar_acceleration(jupiter_position, np.array([0, 0]), jupiter_velocity, M_sun) + compute_polar_acceleration(jupiter_position, third_position, jupiter_velocity, M_third)
    third_velocity += 0.5 * dt * compute_polar_acceleration(third_position, np.array([0, 0]), third_velocity, M_sun) + compute_polar_acceleration(third_position, jupiter_position, third_velocity, M_jupiter)

    r_jupiter += jupiter_velocity[0] * dt
    r_third += third_velocity[0] * dt
    theta_jupiter += jupiter_velocity[1] * dt / r_jupiter
    theta_third += third_velocity[1] * dt / r_third

    acc_jupiter = compute_polar_acceleration(jupiter_position, np.array([0, 0]), jupiter_velocity, M_sun) + compute_polar_acceleration(jupiter_position, third_position, jupiter_velocity, M_third)
    acc_third = compute_polar_acceleration(third_position, np.array([0, 0]), third_velocity, M_sun) + compute_polar_acceleration(third_position, jupiter_position, third_velocity, M_jupiter)

    jupiter_velocity += 0.5 * dt * acc_jupiter
    third_velocity += 0.5 * dt * acc_third

    tot_energy = total_energy(jupiter_position, jupiter_velocity, third_position, third_velocity)
    tot_angular_momentum = total_angular_momentum(jupiter_position, jupiter_velocity, third_position, third_velocity)
    eccentricity_jupiter = compute_eccentricity(jupiter_position, jupiter_velocity, mu_sun)
    eccentricity_third = compute_eccentricity(third_position, third_velocity, mu_sun)

    r_pos_jupiter.append(r_jupiter)
    theta_pos_jupiter.append(theta_jupiter)
    r_pos_third.append(r_third)
    theta_pos_third.append(theta_third)

    total_energy_array.append(tot_energy)
    total_moment_array.append(tot_angular_momentum)
    ecc_jupiter.append(eccentricity_jupiter)
    ecc_third.append(eccentricity_third)
    time_array.append(t)
    t += dt

# Plot 2D trajectory
plt.figure()
plt.plot(r_pos_jupiter * np.cos(theta_pos_jupiter), r_pos_jupiter * np.sin(theta_pos_jupiter), label='Jupiter')
plt.plot(r_pos_third * np.cos(theta_pos_third), r_pos_third * np.sin(theta_pos_third), label='Third body')
plt.scatter([0], [0], color='yellow', label='Sun')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Jupiter and third body 2D trajectory')
plt.gca().set_aspect('equal')
plt.grid()
plt.show()

# Plot eccentricity over time
plt.figure()
plt.plot(np.divide(time_array, 86400), ecc_jupiter)
plt.xlabel('Time (days)')
plt.ylabel('Eccentricity')
plt.title('Jupiter eccentricity over time')
plt.grid()
plt.show()

plt.figure()
plt.plot(np.divide(time_array, 86400), ecc_third)
plt.xlabel('Time (days)')
plt.ylabel('Eccentricity')
plt.title('Third body eccentricity over time')
plt.grid()
plt.show()

# Plot total energy over time
plt.figure()
plt.plot(np.divide(time_array, 86400), total_energy_array)
plt.xlabel('Time (days)')
plt.ylabel('Total Energy')
plt.title('Total Energy over Time')
plt.grid()
plt.show()

# Plot total angular momentum over time
plt.figure()
plt.plot(np.divide(time_array, 86400), total_moment_array)
plt.xlabel('Time (days)')
plt.ylabel('Total Angular Momentum')
plt.title('Total Angular Momentum over Time')
plt.grid()
plt.show()