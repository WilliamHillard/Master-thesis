import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Gravitational Constant (m^3 kg^-1 s^-2)
G = 6.67430e-11

# Sun's Mass (kg)
M_sun = 1.989e30
mu_sun = G * M_sun

# Jupiter's Mass (kg)
M_jupiter = 1.898e27

# Jupiter's Orbital Elements
a_jupiter = 7.7834276e11  # Semi-major axis (m)
e_jupiter = 0.0484  # Eccentricity
i_jupiter = np.radians(1.304)  # Inclination (radians)
omega_jupiter = np.radians(274.3)  # Argument of periapsis (radians)
Omega_jupiter = np.radians(100.4)  # Longitude of ascending node (radians)
theta_jupiter = np.radians(0)  # True anomaly at epoch (radians)


# Function to Convert Orbital Elements to Cartesian Coordinates
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

    x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_prime + (
            -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y_prime
    y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_prime + (
            -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y_prime
    z = (sin_i * sin_omega) * x_prime + (sin_i * cos_omega) * y_prime

    vx = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * v_r + (
            -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * v_theta
    vy = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * v_r + (
            -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * v_theta
    vz = (sin_i * sin_omega) * v_r + (sin_i * cos_omega) * v_theta

    return np.array([x, y, z]), np.array([vx, vy, vz])


# Function to Compute Orbital Energy
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


# Function to Compute Semi-Major Axis
def compute_semi_major_axis(r, v, mu):
    epsilon = orbital_energy(r, v, mu)
    return -mu / (2 * epsilon)


# Convert Jupiter's Orbital Elements to Cartesian Coordinates
jupiter_position, jupiter_velocity = orbital_elements_to_cartesian(a_jupiter, e_jupiter, i_jupiter, omega_jupiter, Omega_jupiter, theta_jupiter, mu_sun)

# Loop test
time_steps = 4380
dt = 86400 # 1 day
eccentricities = []
time_array = []
x_pos = []
y_pos = []
t = 0
r = jupiter_position
v = jupiter_velocity

# Simulation loop
for step in range(time_steps):
    acc = -mu_sun * r / np.linalg.norm(r) ** 3  # Newton's equation of motion
    v += acc * dt  # Update velocity
    r += v * dt  # Update position

    energy = orbital_energy(r, v, mu_sun)
    h = orbital_angular_momentum(r, v)
    e = compute_eccentricity(r, v, mu_sun)
    a = compute_semi_major_axis(r, v, mu_sun)
    eccentricities.append(e)
    x_pos.append(r[0])
    y_pos.append((r[1]))
    time_array.append(t)
    t += dt

# Plot eccentricity over time
plt.figure()
plt.plot(np.divide(time_array, 86400), eccentricities)
plt.xlabel('Time (days)')
plt.ylabel('Eccentricity')
plt.title('Eccentricity over time')
plt.grid()
plt.show()

# Plot 2D trajectory
plt.figure()
plt.plot(x_pos, y_pos, label='Jupiter trajectory')
plt.scatter([0], [0], color='yellow', label='Sun')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Jupiter 2D trajectory')
plt.axis('equal')
plt.grid()
plt.show()