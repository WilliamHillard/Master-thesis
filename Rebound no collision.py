import rebound
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# Constants
G = 6.67430e-11
AU = 1.496e11
M_sun = 1.989e30
M_jupiter = 1.898e27
R_jupiter = 6.9911e7

# Define parameter grid
mass_values = np.linspace(0.5 * M_jupiter, 20 * M_jupiter, 40)
a_values = np.linspace(1 * AU, 5 * AU, 40)
e_values = [0.7]

# 3D array to store final values
results_e_jupiter = np.zeros((len(mass_values), len(a_values), len(e_values)))
results_e_third = np.zeros((len(mass_values), len(a_values), len(e_values)))
results_a_jupiter = np.zeros((len(mass_values), len(a_values), len(e_values)))
results_a_third = np.zeros((len(mass_values), len(a_values), len(e_values)))

time_steps = 200000
dt = 864000

# Loop over grid
for i, M_third in enumerate(mass_values):
    for j, a_third in enumerate(a_values):
        for k, e_third in enumerate(e_values):
            # Create new simulation instance
            sim = rebound.Simulation()
            sim.G = G

            # Add bodies to simulation
            sim.add(m=M_sun, x=0, y=0, z=0, vx=0, vy=0, vz=0)
            sim.add(m=M_jupiter, a=7.7834276e11, e=0.0484, inc=0, Omega=0, omega=0, f=0)
            sim.add(m=M_third, a=a_third, e=e_third, inc=0, Omega=0, omega=0, f=0)

            sim.move_to_com()
            sim.integrator = "ias15"
            sim.dt = dt
            sim.collision = "direct"
            sim.exit_min_distance = 5 * R_jupiter  # Raise error if planets get too close

            jupiter = sim.particles[1]
            third = sim.particles[2]

            try:
                for step in range(time_steps):
                    sim.integrate(sim.t + sim.dt)

                # Store results only if no collision occurred
                results_e_jupiter[i, j, k] = jupiter.e
                results_e_third[i, j, k] = third.e
                results_a_jupiter[i, j, k] = jupiter.a
                results_a_third[i, j, k] = third.a
                print(f"OK: Mass {M_third:.2e}, a = {a_third / AU:.2f} AU, e = {e_third:.2f}")

            except rebound.Encounter:
                print(f"SKIPPED (Collision): Mass {M_third:.2e}, a = {a_third / AU:.2f} AU, e = {e_third:.2f}")
                results_e_jupiter[i, j, k] = np.nan
                results_e_third[i, j, k] = np.nan
                results_a_jupiter[i, j, k] = np.nan
                results_a_third[i, j, k] = np.nan

# Convert values
mass_Mjup = np.array(mass_values) / M_jupiter
a_AU = np.array(a_values) / AU
a_AU_jupiter = results_a_jupiter / AU
a_AU_third = results_a_third / AU

# Mask for unbound orbits (e > 1)
a_AU_jupiter_masked = np.copy(a_AU_jupiter)
a_AU_jupiter_masked[results_e_jupiter >= 1] = np.nan  # Set unbound orbits to NaN

a_AU_third_masked = np.copy(a_AU_third)
a_AU_third_masked[results_e_third >= 1] = np.nan  # Set unbound orbits to NaN

# Identify unbound regions for black overlay
unbound_jupiter = np.copy(a_AU_jupiter)
unbound_jupiter[results_e_jupiter < 1] = np.nan  # Only keep unbound regions

unbound_third = np.copy(a_AU_third)
unbound_third[results_e_third < 1] = np.nan  # Only keep unbound regions

# Clipping values greater than 1
max_ecc = 1  # Anything above this is "unbound"
results_clipped_jupiter = np.clip(results_e_jupiter, 0, max_ecc)
results_main_jupiter = np.copy(results_e_jupiter)
results_main_jupiter[results_e_jupiter >= 1] = np.nan  # Mark unbound as NaN for normal colormap
results_black_jupiter = np.copy(results_e_jupiter)
results_black_jupiter[results_e_jupiter < 1] = np.nan  # Keep only unbound orbits

results_clipped_third = np.clip(results_e_third, 0, max_ecc)
results_main_third = np.copy(results_e_third)
results_main_third[results_e_third >= 1] = np.nan  # Mark unbound as NaN for normal colormap
results_black_third = np.copy(results_e_third)
results_black_third[results_e_third < 1] = np.nan  # Keep only unbound orbits

# Create main colormap (plasma, 0 to 1)
cmap_main = plt.get_cmap("plasma")
norm_main = mcolors.Normalize(vmin=0, vmax=1)

fixed_e_index = 0
fig, ax = plt.subplots(figsize=(8, 6))

# Main color image
im_main = ax.imshow(results_main_jupiter[:, :, fixed_e_index],
                    extent=[a_AU[0], a_AU[-1], mass_Mjup[0], mass_Mjup[-1]],
                    aspect='auto', origin='lower', cmap=cmap_main, norm=norm_main)

# Black overlay for unbound orbits
im_black = ax.imshow(results_black_jupiter[:, :, fixed_e_index],
                     extent=[a_AU[0], a_AU[-1], mass_Mjup[0], mass_Mjup[-1]],
                     aspect='auto', origin='lower', cmap=mcolors.ListedColormap(['black']),
                     norm=mcolors.Normalize(vmin=1, vmax=2))

ax.set_xlabel('Third Body Initial Semi-Major Axis (AU)')
ax.set_ylabel('Third Body Mass (Jupiter masses)')
ax.set_title(f'Jupiter Final Eccentricity (Initial e = {e_values[fixed_e_index]:.2f})')

# Create a single container for both colorbars
divider = make_axes_locatable(ax)

# Add main colorbar
cax_main = divider.append_axes("right", size='5%', pad=0.2)
cbar_main = fig.colorbar(im_main, cax=cax_main)
cbar_main.set_label("Final Eccentricity")
cbar_main.set_ticks(np.linspace(0, 1, 11))

plt.tight_layout()
plt.show()

# Flatten the arrays and structure the data
data = []
for i, M_third in enumerate(mass_values):
    for j, a_third in enumerate(a_values):
        for k, e_third in enumerate(e_values):
            data.append([
                M_third / M_jupiter,  # Convert mass to Jupiter masses
                a_third / AU,  # Convert initial semi-major axis to AU
                e_third,  # Initial eccentricity
                results_e_jupiter[i, j, k],  # Jupiter's final eccentricity
                results_a_jupiter[i, j, k] / AU,  # Jupiter's final semi-major axis in AU
                results_e_third[i, j, k],  # Third body's final eccentricity
                results_a_third[i, j, k] / AU,  # Third body's final semi-major axis in AU
            ])

# Create a DataFrame
df = pd.DataFrame(data, columns=[
    "Third Body Mass (Mjup)", "Initial a (AU)", "Initial e",
    "Jupiter Final e", "Jupiter Final a (AU)",
    "Third Final e", "Third Final a (AU)"
])

# Save to CSV
df.to_csv("0_ecc_no_collisions.csv", index=False)

df = pd.read_csv("0_ecc_no_collisions.csv")