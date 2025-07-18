from src.ParticlesField import ParticlesField
from src.piv_domain import PIVDomain
from src.velocity_field import VelocityField
from src.move_particles import move_particles
from src.correlation import calculate_displacement
import matplotlib.pyplot as plt
import numpy as np

domain1 = PIVDomain(256,256, 0.01)

particles_field1 = ParticlesField(domain1)
image_ti = particles_field1.create_image()

velocity_field1 = VelocityField(domain1.height, domain1.width)
velocity_field1.generate_velocity_field(u_velocity=3, v_velocity=0)
print(velocity_field1)

delta_t=5.0

# calculations positions
new_positions = move_particles(particles_field1.positions, velocity_field1, delta_t)

image_tf = np.zeros((domain1.height, domain1.width), dtype=np.uint8)
for x, y in new_positions:
    # new positions from domain data
    x = int(np.clip(round(x), 0, domain1.width - 1))
    y = int(np.clip(round(y), 0, domain1.height - 1))
    ####
    image_tf[int(y), int(x)] = 255

window=16
overlap=0.5

positions, displacements = calculate_displacement(image_ti, image_tf, window, overlap)

ny = int(np.sqrt(len(positions)))  
nx = len(positions) // ny          

nx * ny == len(positions)

positions = np.array(positions)
displacements = np.array(displacements)

velocities = displacements / delta_t

X = positions[:, 0]
Y = positions[:, 1]
U = displacements[:, 0]
V = displacements[:, 1]

# U = velocities[:, 0]
# V = velocities[:, 1]

#sumarize
print("dx:", np.unique(U))
print("dy:", np.unique(V))

# plt.figure(figsize=(6, 6))
# plt.imshow(image_ti, cmap='gray')  #
# plt.quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=1)
# plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_ti, cmap='gray')
axs[0].set_title('Frame t0')
axs[0].axis('off')

axs[1].imshow(image_tf, cmap='gray')
axs[1].set_title('Frame t1')
axs[1].axis('off')

plt.tight_layout()
plt.show()

#################Velocity Contour###################
#vector magnitude
M = np.sqrt(U**2 + V**2)

print("u:", np.unique(U))
print(" v:", np.unique(V))
print("Max v:", np.max(M))
print("Min v", np.min(M))

#vector grid
X_grid = X.reshape((ny, nx))
Y_grid = Y.reshape((ny, nx))
M_grid = M.reshape((ny, nx))


plt.figure(figsize=(6, 6))
plt.imshow(image_ti, cmap='gray')  
plt.quiver(X, Y, U, V, M, cmap='jet', scale_units='xy', scale=1)
plt.colorbar(label='Displacement [px]')
plt.title("Displacement field")
plt.axis('off')
plt.show()


print("Velocity:", M[:10])
print("Max:", np.max(M))
print("Min:", np.min(M))


X_grid = X.reshape((ny, nx))
Y_grid = Y.reshape((ny, nx))
M_grid = M.reshape((ny, nx))  

# graph velocity
plt.figure(figsize=(6, 5))
plt.contourf(X_grid, Y_grid, M_grid, levels=20, cmap='jet')
plt.title("Velocity contour")
plt.colorbar(label='Velocity')
plt.axis('equal')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()