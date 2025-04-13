import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
num_holes = 100
depth_per_hole = np.arange(0, 50, 0.1)
n_depth = len(depth_per_hole)
surface_elevation = 1500

# Generate grid of coordinates
grid_size = int(np.sqrt(num_holes))
x_vals = np.linspace(499_900, 500_100, grid_size)
y_vals = np.linspace(4_999_900, 5_000_100, grid_size)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
hole_coords = np.c_[x_grid.ravel(), y_grid.ravel()]

# Create 3D geological field
def create_geological_field(x_range, y_range, z_range, resolution=20):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    field = (
        0.5 * np.sin(0.0005 * xx) + 
        0.5 * np.cos(0.0005 * yy) +
        0.3 * np.sin(0.01 * zz) +
        0.2 * np.cos(0.0002 * (xx + yy)) +
        0.1 * np.random.randn(*xx.shape)
    )
    
    field += 0.3 * np.sin(0.002 * xx) * np.cos(0.002 * yy) * np.sin(0.05 * zz)
    
    return RegularGridInterpolator((x, y, z), field, bounds_error=False, fill_value=None)

geo_field = create_geological_field(
    x_range=(499_850, 500_150),
    y_range=(4_999_850, 5_000_150),
    z_range=(surface_elevation-50, surface_elevation)
)

# Container
all_data = []

# Simulate each borehole with lithology-dependent trends
for hole_id, (x0, y0) in enumerate(hole_coords):
    coord_x = np.full(n_depth, x0 + np.random.uniform(-1, 1))
    coord_y = np.full(n_depth, y0 + np.random.uniform(-1, 1))
    coord_z = surface_elevation - depth_per_hole
    
    sample_points = np.column_stack((coord_x, coord_y, coord_z))
    geo_values = geo_field(sample_points)
    geo_values = (geo_values - geo_values.min()) / (geo_values.max() - geo_values.min()) * 1.0 + 0.5
    
    # Enhanced lithology-dependent rock hardness
    lith_hardness = {
        0: 0.8,  # Softest lithology (e.g., shale)
        1: 1.2,  # Medium hardness (e.g., sandstone)
        2: 1.6   # Hardest lithology (e.g., granite)
    }
    
    # Depth-dependent hardness modifier
    depth_hardness = 1 + 0.15 * depth_per_hole / 50  # Increases with depth
    
    # Determine lithology first (needed for hardness calculation)
    lith_threshold1 = 0.8 + 0.1 * np.sin(0.1 * depth_per_hole)
    lith_threshold2 = 1.2 + 0.1 * np.cos(0.08 * depth_per_hole)
    
    lithology = np.zeros(n_depth)
    lithology[geo_values > lith_threshold1] = 1
    lithology[geo_values > lith_threshold2] = 2
    
    # Smooth lithology transitions
    for i in range(1, n_depth-1):
        if np.random.rand() < 0.1:
            lithology[i] = np.random.choice([0,1,2])
        if lithology[i] != lithology[i-1] and lithology[i] != lithology[i+1]:
            lithology[i] = lithology[i-1]
    
    # Calculate composite hardness factor
    composite_hardness = geo_values * depth_hardness * np.array([lith_hardness[l] for l in lithology])
    
    # Drilling parameters with lithology-dependent trends
    # WOB: Increases with hardness but decreases with depth (as drill bit wears)
    wob_base = 8_000 + 4_000 * composite_hardness
    wob = np.random.normal(
        wob_base * (1 - 0.2 * depth_per_hole/50),  # Linear decrease with depth
        300
    )
    
    # RPM: Decreases with hardness, increases with depth (compensation)
    rpm_base = 150 - 30 * composite_hardness
    rpm = np.random.normal(
        rpm_base * (1 + 0.1 * depth_per_hole/50),  # Linear increase with depth
        4
    )
    
    # TRQ: Increases with hardness and depth
    trq_base = 2_000 + 1_000 * composite_hardness
    trq = np.random.normal(
        trq_base * (1 + 0.3 * depth_per_hole/50),  # Strong linear increase
        150
    )
    
    # ROP: Decreases with hardness, increases with WOB/RPM but with diminishing returns
    rop_base = 2.0 / composite_hardness
    rop = np.clip(
        np.random.normal(
            rop_base * (1 + 0.1 * wob/wob_base + 0.05 * rpm/rpm_base),  # Improves with WOB/RPM
            0.1
        ),
        0.05, 3.0  # Reasonable physical limits
    )
    
    # Recalculate SED with new parameters
    sed = (2 * np.pi * trq * rpm + wob * rop) / rop
    sed += np.random.normal(0, 30_000, n_depth)
    
    # Build DataFrame
    hole_df = pd.DataFrame({
        'HoleID': hole_id,
        'CoordX': coord_x,
        'CoordY': coord_y,
        'CoordZ': coord_z,
        'Depth': depth_per_hole,
        'WOB': wob,
        'RPM': rpm,
        'TRQ': trq,
        'ROP': rop,
        'SED': sed,
        'Lithology': lithology,
        'Hardness': composite_hardness  # Added hardness for analysis
    })

    all_data.append(hole_df)

# Final dataset
df = pd.concat(all_data, ignore_index=True)

# Save to CSV
df.to_csv('drilling_data.csv', index=False)
print("Data saved to drilling_data_with_lithology_trends.csv")

# Create 3D plot of SED
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Sample every 5th point for better visualization
plot_df = df.iloc[::5]

# Scatter plot with color mapping
sc = ax.scatter(
    plot_df['CoordX'], 
    plot_df['CoordY'], 
    plot_df['CoordZ'], 
    c=plot_df['SED'],
    cmap='viridis',
    alpha=0.6,
    s=5
)

# Labels and title
ax.set_xlabel('Coord X (m)')
ax.set_ylabel('Coord Y (m)')
ax.set_zlabel('Elevation (m)')
ax.set_title('3D Distribution of SED (J/m³) with Lithology-Dependent Trends')

# Color bar
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('SED (J/m³)')

plt.tight_layout()
plt.savefig('3d_sed_distribution_with_trends.png', dpi=300)
plt.show()

# Additional plot: ROP vs Depth colored by Lithology
plt.figure(figsize=(10, 6))
for lith, color in zip([0, 1, 2], ['green', 'orange', 'red']):
    subset = df[df['Lithology'] == lith].iloc[::10]
    plt.scatter(subset['Depth'], subset['ROP'], c=color, label=f'Lithology {lith}', alpha=0.5)

plt.xlabel('Depth (m)')
plt.ylabel('ROP (m/min)')
plt.title('ROP vs Depth by Lithology')
plt.legend()
plt.grid(True)
plt.savefig('rop_vs_depth_by_lithology.png', dpi=300)
plt.show()