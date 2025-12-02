"""
Generate a background image for the hero section based on research themes:
- Wave propagation (acoustic waves)
- Computational fluid dynamics (flow patterns)
- Finite element meshes
- Mathematical optimization (gradient flows)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.patheffects as path_effects

# Set high resolution
plt.figure(figsize=(16, 9), dpi=150)
ax = plt.gca()

# Create a gradient background (blue tones matching the website)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Gradient from deep blue to lighter blue
color1 = np.array([37, 99, 235]) / 255  # #2563eb
color2 = np.array([14, 165, 233]) / 255  # #0ea5e9

R = color1[0] * (1 - Y) + color2[0] * Y
G = color1[1] * (1 - Y) + color2[1] * Y
B = color1[2] * (1 - Y) + color2[2] * Y

background = np.dstack([R, G, B])
ax.imshow(background, extent=[0, 10, 0, 10], aspect='auto', alpha=0.9)

# Add wave patterns (representing acoustic waves and CFD)
t = np.linspace(0, 10, 1000)
for i, (freq, phase, alpha_val) in enumerate([
    (2, 0, 0.15),
    (3, np.pi/4, 0.12),
    (4, np.pi/2, 0.10),
    (2.5, np.pi, 0.08)
]):
    y_wave = 5 + 2 * np.sin(freq * t + phase)
    ax.plot(t, y_wave, 'white', linewidth=2, alpha=alpha_val)
    # Add harmonics
    y_wave2 = 5 + 1.5 * np.sin(freq * t * 1.5 + phase)
    ax.plot(t, y_wave2, 'white', linewidth=1.5, alpha=alpha_val * 0.6)

# Add finite element mesh representation (triangulation)
np.random.seed(42)
n_points = 25
mesh_points_x = np.random.uniform(0, 10, n_points)
mesh_points_y = np.random.uniform(0, 10, n_points)

from matplotlib.tri import Triangulation
triang = Triangulation(mesh_points_x, mesh_points_y)

# Draw mesh with low opacity
ax.triplot(triang, color='white', linewidth=0.5, alpha=0.15)

# Add some gradient flow arrows (representing optimization/adjoint methods)
Y_grid, X_grid = np.mgrid[1:9:8j, 1:9:8j]

# Create a simple vector field (gradient-like)
U = -0.3 * (X_grid - 5)
V = -0.3 * (Y_grid - 5)

ax.quiver(X_grid, Y_grid, U, V, 
          color='white', alpha=0.2, width=0.003, 
          scale=8, headwidth=4, headlength=5)

# Add subtle mathematical symbols/equations in the background
equations = [
    (r'$\nabla \cdot u = 0$', (1.5, 8.5)),  # Incompressibility
    (r'$\frac{\partial u}{\partial t} + u \cdot \nabla u = -\nabla p + \nu \nabla^2 u$', (7, 1.5)),  # Navier-Stokes
    (r'$\nabla^2 p + \omega^2 \rho p = 0$', (1, 2)),  # Wave equation
    (r'$\min_{m} J(m)$', (8.5, 8)),  # Optimization
    (r'$\frac{dJ}{dm} = \nabla_m J$', (2.5, 2.5)),  # Gradient
]

for eq, (x_pos, y_pos) in equations:
    text = ax.text(x_pos, y_pos, eq, fontsize=14, color='white', 
                   alpha=0.15, family='serif',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='none', 
                            edgecolor='none'))

# Add some circular elements (representing computational domains)
circles_data = [
    (2, 7, 0.6, 0.08),
    (8, 3, 0.5, 0.1),
    (5, 5, 0.8, 0.06),
]

for cx, cy, radius, alpha_val in circles_data:
    circle = Circle((cx, cy), radius, fill=False, 
                   edgecolor='white', linewidth=2, alpha=alpha_val)
    ax.add_patch(circle)

# Clean up axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Save with transparent edges
plt.tight_layout(pad=0)
plt.savefig('hero-background.png', dpi=150, bbox_inches='tight', 
            pad_inches=0, facecolor='none', edgecolor='none')
plt.savefig('hero-background.jpg', dpi=150, bbox_inches='tight', 
            pad_inches=0, facecolor='#2563eb')

print("Background images generated successfully!")
print("- hero-background.png (with transparency)")
print("- hero-background.jpg (with solid blue background)")
