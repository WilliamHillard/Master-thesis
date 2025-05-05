import rebound
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
G = 6.67430e-11
AU = 1.496e11
M_sun = 1.989e30
M_jupiter = 1.898e27
M_third = 5.5 * M_jupiter
R_jupiter = 7.1492e7

# Create simulation
sim = rebound.Simulation()
sim.G = G  # Set gravitational constant

# Add bodies to simulation
sim.add(m=M_sun, x=0, y=0, z=0, vx=0, vy=0, vz=0)
sim.add(m=M_jupiter, a=5.20282593583 * AU, e=0.0484, inc=0, Omega=0, omega=0, f=0)
sim.add(m=M_third, a=7.55 * AU, e=0.0, inc=0, Omega=0, omega=0, f=0)

sim.move_to_com()  # Center of mass in the origin

sim.integrator = "ias15"
sim.dt = 864000
sim.collision = "direct"
sim.exit_min_distance = 4 * R_jupiter

time_steps = 4000000000
times = []
x_jupiter = []
y_jupiter = []
x_third = []
y_third = []
x_sun = []
y_sun = []
ecc_jupiter = []
ecc_third = []
a_jupiter = []
a_third = []
total_energy = []
total_angular_momentum = []

# Extract particles
sun, jupiter, third = sim.particles[0], sim.particles[1], sim.particles[2]

save_interval = 10000

# Run simulation
try:
    for step in range(time_steps):
        sim.integrate(sim.t + sim.dt)

        if step % save_interval == 0:  # Only save every 1000th step

            x_jupiter.append(jupiter.x / 149597870700)
            y_jupiter.append(jupiter.y / 149597870700)
            x_third.append(third.x / 149597870700)
            y_third.append(third.y / 149597870700)
            x_sun.append(sun.x / 149597870700)
            y_sun.append(sun.y / 149597870700)
            ecc_jupiter.append(jupiter.e)
            ecc_third.append(third.e)
            a_jupiter.append(jupiter.a / 149597870700)
            a_third.append(third.a / 149597870700)
            times.append(sim.t / 31536000)  # Store time in days

except rebound.Encounter as err:
    collision_time_years = sim.t / 31536000
    print(f"Simulation stopped early due to close encounter: {err}")
    print(f"Collision occurred at {collision_time_years:.2f} years")

# Plots
plt.figure()
plt.plot(times, ecc_jupiter, label="Jupiter Eccentricity")
plt.xlabel("Time (years)")
plt.ylabel("Eccentricity")
plt.title("Jupiter eccentricity over time")
plt.grid()
plt.show()

plt.figure()
plt.plot(times, ecc_third, label="Third Body Eccentricity")
plt.xlabel("Time (years)")
plt.ylabel("Eccentricity")
plt.title("Third body eccentricity over time")
plt.grid()
plt.show()

plt.figure()
plt.plot(times, a_jupiter, label="Jupiter Semi-major axis")
plt.xlabel("Time (years)")
plt.ylabel("Semi-major axis (AU)")
plt.title("Jupiter semi-major axis over time")
plt.grid()
plt.show()

plt.figure()
plt.plot(times, a_third, label="Third body Semi-major axis")
plt.xlabel("Time (years)")
plt.ylabel("Semi-major axis (AU)")
plt.title("Third body semi-major axis over time")
plt.grid()
plt.show()

plt.figure()
plt.plot(x_jupiter, y_jupiter, label="Jupiter", color='blue')
plt.plot(x_third, y_third, label="Third Body", color='red')
plt.plot(x_sun, y_sun, label="Sun", color='orange')
plt.xlabel("X Position (AU)")
plt.ylabel("Y Position (AU)")
plt.title("Jupiter and third body 2D trajectory")
plt.grid()
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

# Save data
# Convert lists to arrays
x_jupiter = np.array(x_jupiter)
y_jupiter = np.array(y_jupiter)
x_third = np.array(x_third)
y_third = np.array(y_third)
ecc_jupiter = np.array(ecc_jupiter)
ecc_third = np.array(ecc_third)
times = np.array(times)

# Create a DataFrame
df = pd.DataFrame({
    "Time (years)": times,
    "Jupiter x (AU)": x_jupiter,
    "Jupiter y AU)": y_jupiter,
    "Third x (AU)": x_third,
    "Third y (AU)": y_third,
    "Jupiter e": ecc_jupiter,
    "Third e": ecc_third
})

# Save to CSV
df.to_csv("extralong_case6_ias15.csv", index=False)

print("CSV file saved as 'extralong_case6_ias15.csv'")