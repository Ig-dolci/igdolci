"""
Generate a sophisticated background image for the hero section based on research themes:
- Wave propagation (acoustic waves)
- Computational fluid dynamics (flow patterns)
- Finite element meshes
- Mathematical optimization (gradient flows)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation

# Set dimensions for very wide display (4:1 aspect ratio)
fig = plt.figure(figsize=(48, 12), dpi=200)
ax = plt.gca()

# Create sophisticated multi-layer gradient background
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)

# Define color palette - dark teal to turquoise
color_dark = np.array([15, 76, 92]) / 255   # Deep ocean teal
color_mid = np.array([20, 184, 166]) / 255  # Teal
color_light = np.array([94, 234, 212]) / 255  # Bright turquoise

# Create layered gradient with radial influences
R = (color_dark[0] * (1 - Y) + color_light[0] * Y) * (1 - 0.2 * X)
G = (color_dark[1] * (1 - Y) + color_light[1] * Y) * (1 - 0.15 * X)
B = (color_dark[2] * (1 - Y) + color_light[2] * Y) * (1 - 0.1 * X)

# Add subtle radial gradients for depth
center_x, center_y = 0.3, 0.6
radial = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
radial_factor = 1 - 0.15 * np.exp(-radial * 3)

background = np.dstack([R * radial_factor, G * radial_factor, B * radial_factor])
ax.imshow(background, extent=[0, 40, 0, 10], aspect='auto', alpha=1.0)

# Create elegant finite element mesh - structured regions
np.random.seed(42)

# Create multiple mesh regions for visual interest
mesh_regions = [
    (np.array([1, 8, 1, 4]), 20),         # Left
    (np.array([32, 39, 6, 9]), 18),       # Right
    (np.array([15, 25, 3, 7]), 25),       # Center
]

for bounds, n_pts in mesh_regions:
    mesh_x = np.random.uniform(bounds[0], bounds[1], n_pts)
    mesh_y = np.random.uniform(bounds[2], bounds[3], n_pts)
    triang = Triangulation(mesh_x, mesh_y)
    ax.triplot(triang, color='white', linewidth=0.6, alpha=0.12, linestyle='-')

# Vector field removed for cleaner design

# Add elegant mathematical annotations with better positioning
equations = [
    (r'$\nabla \cdot \mathbf{u} = 0$', (3, 8.5), 40),
    (r'$\nabla^2 p = -\omega^2 c^{-2} p$', (33, 1.5), 36),
    (r'$\min_{\mathbf{m}} \, J(\mathbf{m})$', (35, 8.5), 40),
    (r'$\mathbf{g} = \nabla_{\mathbf{m}} J$', (4, 1.8), 36),
]

for eq, (x_pos, y_pos), fsize in equations:
    text = ax.text(x_pos, y_pos, eq, fontsize=fsize, color='white',
                   alpha=0.18, family='serif', weight='normal',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='none',
                            edgecolor='none'))

# Add sophisticated geometric elements
# Large circles representing computational domains
circles = [
    (8, 7.2, 0.8, 0.10, 2.5),
    (32, 3.0, 0.7, 0.12, 2.2),
    (20, 5.2, 1.0, 0.08, 2.8),
]

for cx, cy, radius, alpha_val, lw in circles:
    circle = Circle((cx, cy), radius, fill=False,
                   edgecolor='white', linewidth=lw, alpha=alpha_val)
    ax.add_patch(circle)
    # Add inner circle for depth
    inner_circle = Circle((cx, cy), radius * 0.6, fill=False,
                         edgecolor='white', linewidth=lw * 0.6, alpha=alpha_val * 0.5)
    ax.add_patch(inner_circle)

# Add subtle grid overlay for structure
for i in np.linspace(2, 38, 20):
    ax.plot([i, i], [0, 10], 'white', linewidth=0.3, alpha=0.04)
for i in np.linspace(0.5, 9.5, 10):
    ax.plot([0, 40], [i, i], 'white', linewidth=0.3, alpha=0.04)

# Add decorative corner elements
corner_elements = [
    (0.3, 0.3, 0.5, 0.5),
    (39.7, 9.7, -0.5, -0.5),
]

for cx, cy, dx, dy in corner_elements:
    ax.plot([cx, cx + dx], [cy, cy], 'white', linewidth=3, alpha=0.2,
           solid_capstyle='round')
    ax.plot([cx, cx], [cy, cy + dy], 'white', linewidth=3, alpha=0.2,
           solid_capstyle='round')

# Add subtle particle effects
np.random.seed(123)
n_particles = 150
px = np.random.uniform(0, 40, n_particles)
py = np.random.uniform(0, 10, n_particles)
sizes = np.random.uniform(1, 4, n_particles)
alphas = np.random.uniform(0.05, 0.15, n_particles)

for i in range(n_particles):
    ax.plot(px[i], py[i], 'o', color='white', markersize=sizes[i],
           alpha=alphas[i])

# Clean up
ax.set_xlim(0, 40)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_aspect('auto')

# Save high-quality outputs
plt.tight_layout(pad=0)
plt.savefig('hero-background.png', dpi=200, bbox_inches='tight',
            pad_inches=0, facecolor='none', edgecolor='none',
            transparent=True)
plt.savefig('hero-background.jpg', dpi=200, bbox_inches='tight',
            pad_inches=0, facecolor='#0f4c5c', edgecolor='none')
plt.close()

print("✓ High-quality background images generated successfully!")
print("  - hero-background.png (transparent, 200 DPI)")
print("  - hero-background.jpg (solid background, 200 DPI)")
