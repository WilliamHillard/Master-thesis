import numpy as np
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
i_jupiter = np.radians(0)  # Inclination (radians) 1.304
omega_jupiter = np.radians(0)  # Argument of periapsis (radians) 274.3
Omega_jupiter = np.radians(0)  # Longitude of ascending node (radians) 100.4
theta_jupiter = np.radians(0)  # True anomaly at epoch (radians)

# Earth's Orbital Elements
a_third = 1.496e11
e_third = 0.0167
i_third = np.radians(0)
omega_third = np.radians(0)
Omega_third = np.radians(0)
theta_third = np.radians(0)


def orbital_elements_to_cartesian(a, e, i, omega, Omega, theta, mu):
    """
    Convert orbital elements to position and velocity in Cartesian coordinates.
    a: semi-major axis (m)
    e: eccentricity
    i: inclination (radians)
    omega: argument of periapsis (radians)
    Omega: longitude of ascending node (radians)
    theta: true anomaly (radians)
    mu: gravitational parameter (G * M_sun)
    """
    r = a * (1 - e ** 2) / (1 + e * np.cos(theta))

    # Position in orbital plane
    x_prime = r * np.cos(theta)
    y_prime = r * np.sin(theta)

    # Velocity in orbital plane
    h = np.sqrt(mu * a * (1 - e ** 2))  # Specific angular momentum
    v_r = (mu / h) * e * np.sin(theta)
    v_theta = (mu / h) * (1 + e * np.cos(theta))

    # Rotate to 3D space
    cos_Omega, sin_Omega = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)

    # x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_prime + (
    #        -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y_prime
    # y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_prime + (
    #        -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y_prime
    # z = (sin_i * sin_omega) * x_prime + (sin_i * cos_omega) * y_prime

    # vx = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * v_r + (
    #        -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * v_theta
    # vy = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * v_r + (
    #        -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * v_theta
    # vz = (sin_i * sin_omega) * v_r + (sin_i * cos_omega) * v_theta

    x = x_prime
    y = y_prime
    z = 0

    vx = v_r
    vy = v_theta
    vz = 0

    return np.array([x, y, z]), np.array([vx, vy, vz])


def orbital_energy(r, v, mu):
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


# Function to Compute Orbital Angular Momentum
def orbital_angular_momentum(r, v):
    return np.linalg.norm(np.cross(r, v))


# Function to Compute Eccentricity
def compute_eccentricity(r, v, mu):
    h = orbital_angular_momentum(r, v)
    epsilon = orbital_energy(r, v, mu)
    e = np.sqrt(1 + (2 * epsilon * h ** 2) / (mu ** 2))
    return e


# Function to Compute Total Energy
def total_energy(r_j, v_j, r_t, v_t):
    KE = 0.5 * M_jupiter * np.dot(v_j, v_j) + 0.5 * M_third * np.dot(v_t, v_t)
    PE_sun = -G * M_sun * (M_jupiter / np.linalg.norm(r_j) + M_third / np.linalg.norm(r_t))
    PE_interaction = -G * M_jupiter * M_third / np.linalg.norm(r_j - r_t)
    return KE + PE_sun + PE_interaction


# Function to Compute Total Angular Momentum
def total_angular_momentum(r_j, v_j, r_t, v_t):
    mom_jupiter = M_jupiter * np.cross(r_j, v_j)
    mom_third = M_third * np.cross(r_t, v_t)
    return np.linalg.norm(mom_jupiter + mom_third)


# Function to Compute Acceleration
def compute_acceleration(r1, r2, m2):
    r_vec = r2 - r1
    r_mag = np.linalg.norm(r_vec)
    return G * m2 * r_vec / r_mag ** 3


# Convert Orbital Elements to Cartesian Coordinates
jupiter_position, jupiter_velocity = orbital_elements_to_cartesian(a_jupiter, e_jupiter, i_jupiter, omega_jupiter,
                                                                   Omega_jupiter, theta_jupiter, mu_sun)
#third_position, third_velocity = orbital_elements_to_cartesian(a_third, e_third, i_third, omega_third, Omega_third, theta_third, mu_sun)

# Leapfrog Integration Setup
time_steps = 10000  # step: 10168 vid x=0, 10501 vid y_max=777428176035.929
dt = 86400  # 1 day
t = 0
time_array = []
total_energy_array = []
total_moment_array = []
x_pos_jupiter = []
y_pos_jupiter = []
x_pos_third = []
y_pos_third = []

r_j = jupiter_position
v_j = jupiter_velocity
r_third = np.array([7e11, 0, 0], dtype=np.float64)
v_third = np.array([0, 10000, 0], dtype=np.float64)
M_third = 9.002e27  # Earth 5.972e24

ecc_jupiter = []
ecc_third = []
ener_jupiter = []
moment_jupiter = []
ener_third = []
moment_third = []

# Simulation loop using Leapfrog integration
for step in range(time_steps):
    # Half-step velocity update
    v_j += 0.5 * dt * (
                compute_acceleration(r_j, np.array([0, 0, 0]), M_sun) + compute_acceleration(r_j, r_third, M_third))
    v_third += 0.5 * dt * (
                compute_acceleration(r_third, np.array([0, 0, 0]), M_sun) + compute_acceleration(r_third, r_j,
                                                                                                 M_jupiter))

    # Full position update
    r_j += v_j * dt
    r_third += v_third * dt

    acc_j = compute_acceleration(r_j, np.array([0, 0, 0]), M_sun) + compute_acceleration(r_j, r_third, M_third)
    acc_third = compute_acceleration(r_third, np.array([0, 0, 0]), M_sun) + compute_acceleration(r_third, r_j,
                                                                                                 M_jupiter)

    # Half-step velocity update
    v_j += 0.5 * dt * acc_j
    v_third += 0.5 * dt * acc_third

    e_jupiter = compute_eccentricity(r_j, v_j, mu_sun)
    e_third = compute_eccentricity(r_third, v_third, mu_sun)

    ecc_jupiter.append(e_jupiter)
    ecc_third.append(e_third)
    x_pos_jupiter.append(r_j[0])
    y_pos_jupiter.append(r_j[1])
    x_pos_third.append(r_third[0])
    y_pos_third.append(r_third[1])

    # To see plots (will remove later)
    #energy_jupiter = orbital_energy(r_j, v_j, mu_sun)
    #h_jupiter = orbital_angular_momentum(r_j, v_j)
    #energy_third = orbital_energy(r_third, v_third, mu_sun)
    #h_third = orbital_angular_momentum(r_third, v_third)
    #energy_total = total_energy(r_j, v_j, r_third, v_third)
    #angular_momentum_total = total_angular_momentum(r_j, v_j, r_third, v_third)
    #ener_jupiter.append(energy_jupiter)
    #moment_jupiter.append(h_jupiter)
    #ener_third.append(energy_third)
    #moment_third.append(h_third)
    #total_energy_array.append(energy_total)
    #total_moment_array.append(angular_momentum_total)

    time_array.append(t)
    t += dt

# Plot 2D trajectory
plt.figure()
plt.plot(x_pos_jupiter, y_pos_jupiter, label='Jupiter')
plt.plot(x_pos_third, y_pos_third, label='Third body')
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

# Plot energy over time
plt.figure()
plt.plot(np.divide(time_array, 86400), ener_jupiter)
plt.xlabel('Time (days)')
plt.ylabel('Energy')
plt.title('Jupiter energy over time')
plt.grid()
# plt.show()

plt.figure()
plt.plot(np.divide(time_array, 86400), ener_third)
plt.xlabel('Time (days)')
plt.ylabel('Energy')
plt.title('Third body energy over time')
plt.grid()
# plt.show()

# Plot angular momentum over time
plt.figure()
plt.plot(np.divide(time_array, 86400), moment_jupiter)
plt.xlabel('Time (days)')
plt.ylabel('Angular momentum')
plt.title('Jupiter angular momentum over time')
plt.grid()
# plt.show()

plt.figure()
plt.plot(np.divide(time_array, 86400), moment_third)
plt.xlabel('Time (days)')
plt.ylabel('Angular momentum')
plt.title('Third body angular momentum over time')
plt.grid()
# plt.show()

# Plot total energy over time
plt.figure()
plt.plot(np.divide(time_array, 86400), total_energy_array)
plt.xlabel('Time (days)')
plt.ylabel('Total Energy')
plt.title('Total Energy over Time')
plt.grid()
# plt.show()

# Plot total angular momentum over time
plt.figure()
plt.plot(np.divide(time_array, 86400), total_moment_array)
plt.xlabel('Time (days)')
plt.ylabel('Total Angular Momentum')
plt.title('Total Angular Momentum over Time')
plt.grid()
# plt.show()
