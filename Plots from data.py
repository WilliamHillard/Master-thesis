import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("0_ecc_no_collisions.csv")

# Extract columns
x_data = df.iloc[:, 3]  # Final e
y_data = df.iloc[:, 4]  # Final a
third_mass = df.iloc[:, 0]  # Third body mass
initial_a = df.iloc[:, 1]  # Initial semi-major axis
initial_e = df.iloc[:, 2]  # Initial eccentricity

# Scatter plot with third body mass as colorbar
plt.figure(figsize=(8, 6))
sc = plt.scatter(y_data, x_data, c=third_mass, cmap="inferno", alpha=0.7, edgecolors='k')
cbar = plt.colorbar(sc)
cbar.set_label("Third body mass")
plt.xscale("log")
plt.xlabel("Final Semi-Major Axis (AU)")
plt.ylabel("Final Eccentricity")
plt.title("Semi-Major Axis vs. Eccentricity")
plt.grid()
plt.ylim(0, 1)
plt.show()

# Scatter plot with initial semi-major axis as colorbar
plt.figure(figsize=(8, 6))
sc = plt.scatter(y_data, x_data, c=initial_a, cmap="inferno", alpha=0.7, edgecolors='k')
cbar = plt.colorbar(sc)
cbar.set_label("Initial semi-major axis")
plt.xscale("log")
plt.xlabel("Final Semi-Major Axis (AU)")
plt.ylabel("Final Eccentricity")
plt.title("Semi-Major Axis vs. Eccentricity")
plt.grid()
plt.ylim(0, 1)
plt.show()

# Scatter plot with initial eccentricity as colorbar
plt.figure(figsize=(8, 6))
sc = plt.scatter(y_data, x_data, c=initial_e, cmap="inferno", alpha=0.7, edgecolors='k')
cbar = plt.colorbar(sc)
cbar.set_label("Initial eccentricity")
plt.xscale("log")
plt.xlabel("Final Semi-Major Axis (AU)")  # Change to actual label
plt.ylabel("Final Eccentricity")  # Change to actual label
plt.title("Semi-Major Axis vs. Eccentricity")
plt.grid()
plt.ylim(0, 1)
plt.show()

# How many collisions?
total_simulations = len(df)

# Count simulations with any NaN in relevant final columns
disrupted = df[
    df[[
        "Jupiter Final e",
        "Jupiter Final a (AU)",
        "Third Final e",
        "Third Final a (AU)"
    ]].isna().any(axis=1)
]

num_disrupted = len(disrupted)

# Percentage
fraction_disrupted = num_disrupted / total_simulations * 100

# Output
print(f"Total simulations: {total_simulations}")
print(f"Disrupted/collided simulations: {num_disrupted}")
print(f"Fraction: {fraction_disrupted:.2f}%")