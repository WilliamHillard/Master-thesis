import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# Load data
df = pd.read_csv("t4000000_dt10days_mass0.5-20-40_a6-8-40_ecc0.0.csv")

# Drop rows with NaN or inf in final_a or final_e
df = df[np.isfinite(df.iloc[:, 3]) & np.isfinite(df.iloc[:, 4])]

# Now extract
final_e = df.iloc[:, 3]
final_a = df.iloc[:, 4]
third_mass = df.iloc[:, 0]
initial_a = df.iloc[:, 1]
initial_e = df.iloc[:, 2]

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(final_a, final_e, c=initial_e, cmap="inferno", alpha=0.7, edgecolors='k')
cbar = plt.colorbar(sc)
cbar.set_label("Initial eccentricity")
ax.set_xscale("log")
ax.set_xlabel("Final Semi-Major Axis (AU)")
ax.set_ylabel("Final Eccentricity")
ax.set_title("Semi-Major Axis vs. Eccentricity")
ax.set_xlim(2, 4)
ax.set_ylim(0.7, 0.87)
ax.grid()

# Show plot and click planets
print("Click points on the figure to select planets")
plt.tight_layout()
plt.draw()
clicked_points = plt.ginput(n=-1, timeout=0)  # n=-1: unlimited until Enter is pressed
plt.close()

# Convert clicked points to numpy array
clicked_points = np.array(clicked_points)

# Build KDTree for fast nearest-neighbor matching
tree = cKDTree(np.c_[final_a, final_e])
distances, indices = tree.query(clicked_points)

# Extract selected rows
selected_planets_outside_0ecc = df.iloc[indices]

# Show and save selected initial conditions
print("\nSelected planets:")
print(selected_planets_outside_0ecc.iloc[:, [2, 1, 0]])  # initial a, initial e, third mass

# Optional: save to CSV
selected_planets_outside_0ecc.to_csv("selected_planets_outside_0ecc.csv", index=False)