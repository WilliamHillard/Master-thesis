import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# Use the Planetary Systems (PS) table
data = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="pl_name,pl_orbsmax,pl_orbeccen,pl_bmassj,pl_orbper",
    where="pl_orbsmax IS NOT NULL AND pl_orbeccen IS NOT NULL AND pl_bmassj IS NOT NULL"
)

# Convert data
df = data.to_pandas()
df = df.rename(columns={
    "pl_name": "name",
    "pl_orbsmax": "semi_major_axis",
    "pl_orbeccen": "eccentricity",
    "pl_bmassj": "mass",
    "pl_orbper": "orbital_period"
})

# Filters
filtered = df[
    (df['mass'] > 0.5) &
    (df['mass'] < 10) &
    (df['semi_major_axis'] < 5)
]

# Number of planets
print(len(filtered))

# Extract data for plot
a = filtered['semi_major_axis']
e = filtered['eccentricity']
mass = filtered['mass']
y = e ** 2

# High-eccentricity tidal migration limits
e_zone = np.linspace(0.0, 0.99, 500)
e2_zone = e_zone ** 2
a_min = 0.034 / (1 - e2_zone)
a_max = 0.1 / (1 - e2_zone)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Roche zone
ax.fill_betweenx(e2_zone, a_min, a_max, color='lightgray', alpha=0.5, label='High eccentricity tidal migration', zorder=0)

# Scatter plot
sc = ax.scatter(a, y, c=mass, cmap='viridis', s=40, edgecolor='k', alpha=0.9)

# X-axis
ax.set_xscale('log')
ax.set_xlim(0.01, 5)
ax.set_xticks([0.01, 0.1, 1, 5])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.ticklabel_format(axis='x', style='plain')
ax.set_xlabel("Semi-major Axis [AU]")

# Y-axis
e_ticks = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.set_yticks(np.square(e_ticks))
ax.set_yticklabels([f"{e:.2f}" for e in e_ticks])
ax.set_ylabel("Eccentricity")
ax.set_ylim(0, 1)

# Colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Mass [Mjup]")

# Plot
ax.grid(True, which="both", ls="--", alpha=0.2)
ax.legend()
plt.title("Eccentricity vs Semi-Major Axis")
plt.tight_layout()
plt.show()