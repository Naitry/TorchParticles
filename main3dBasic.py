import glfw
from OpenGL.GL import *
import numpy as np
import torch

screenSize = 2000

# Constants
NumberDimensions = 3
Particles = 1000
N = 2
c = 100000.0  # Set the speed of light or another relativistic speed limit
damping = 0.01

# Coupling matrix
C = torch.tensor([[500., 0.],
                  [0., 0.]])

# Particle properties
particle_properties = torch.ones(Particles, N)

# Add repulsive force at the walls
wall_force_strength = -500
wall_force_thickness = 0.1

central_force_strength = -200.0
    
# Simulation loop
timesteps = 50
dt = 0.00005

def draw_particles(positions):
    N = positions.shape[0]

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_POINTS)
    for pos in positions:
        x, y, z = pos
        glVertex2f(x / (z + 2), y / (z + 2))
    glEnd()

    
def wall_forces(positions, wall_force_strength, wall_force_thickness):
    left_wall = positions[:, 0] + 1
    right_wall = 1 - positions[:, 0]
    bottom_wall = positions[:, 1] + 1
    top_wall = 1 - positions[:, 1]
    front_wall = positions[:, 2] + 1
    back_wall = 1 - positions[:, 2]

    force_x = -wall_force_strength * (1 / (left_wall + wall_force_thickness) - 1 / wall_force_thickness) \
              + wall_force_strength * (1 / (right_wall + wall_force_thickness) - 1 / wall_force_thickness)
    force_y = -wall_force_strength * (1 / (bottom_wall + wall_force_thickness) - 1 / wall_force_thickness) \
              + wall_force_strength * (1 / (top_wall + wall_force_thickness) - 1 / wall_force_thickness)
    force_z = -wall_force_strength * (1 / (front_wall + wall_force_thickness) - 1 / wall_force_thickness) \
              + wall_force_strength * (1 / (back_wall + wall_force_thickness) - 1 / wall_force_thickness)

    wall_forces = torch.stack((force_x, force_y, force_z), dim=-1)
    return wall_forces

    
def central_force(positions, central_force_strength):
    # Calculate the radial squared distance from the origin
    radial_squared = (positions ** 2).sum(dim=-1)

    # Calculate the force magnitude for each particle
    force_magnitude = central_force_strength * radial_squared * screenSize

    # Calculate the normalized direction of the force for each particle
    force_direction = positions / (torch.sqrt(radial_squared) + 1e-9).unsqueeze(-1)

    # Compute the force vectors
    force_vectors = force_magnitude.unsqueeze(-1) * force_direction

    return force_vectors

def simulate(positions, velocities, particle_properties, C, dt):
    # Calculate pairwise distances and displacement vectors
    displacement_vectors =  positions[None, :, :] - positions[:, None, :]
    distances = torch.norm(displacement_vectors, dim=2, keepdim=True)
    distances += torch.eye(Particles)[:, :, None]  # Add identity to avoid division by zero

    # Calculate interaction values and force vectors
    interaction_values = torch.matmul(torch.matmul(particle_properties[:, None, :], C), particle_properties[None, :, :].transpose(-1, -2)).diagonal(dim1=1, dim2=2)
    force_vectors = interaction_values[:, :, None] * displacement_vectors / distances ** 3

    # Remove self-interaction
    mask = torch.ones(Particles, Particles, dtype=bool) ^ torch.eye(Particles, dtype=bool)
    force_vectors = force_vectors[mask].reshape(Particles, Particles - 1, NumberDimensions)

    # Calculate total forces and accelerations
    total_forces = torch.sum(force_vectors, dim=1)
    total_forces += central_force(positions, central_force_strength)
    total_forces = torch.clamp(total_forces, min=-1e8, max=1e8)


    epsilon = 1e-9
    relativistic_factor = (1 - (velocities ** 2).sum(dim=-1) / c ** 2 + epsilon) ** (-3/2)


    accelerations = total_forces / particle_properties[:, 0:1] * relativistic_factor.unsqueeze(-1) * damping

    # Update positions and velocities
    positions += velocities * dt + 0.5 * accelerations * dt**2
    velocities += accelerations * dt

    return positions, velocities


import numpy as np

def initialize_positions(Particles):
    # Generate random angles
    angles = np.random.uniform(0, 2 * np.pi, (Particles, 1))
    phi = np.random.uniform(0, np.pi, (Particles, 1))

    # Generate random radii from a normal distribution
    radii = np.random.normal(loc=0, scale=0.5, size=(Particles, 1))

    # Compute the x, y, and z coordinates from the angles and radii
    x = radii**2 * np.sin(phi) * np.cos(angles)
    y = radii**2 * np.sin(phi) * np.sin(angles)
    z = radii**2 * np.cos(phi)

    # Combine x, y, and z coordinates
    positions = torch.tensor(np.hstack((x, y, z)), dtype=torch.float32)

    # Calculate the initial velocities
    axis = torch.tensor([0.0, 1.0, 0.])
    velocities = torch.cross(positions, axis.repeat(Particles, 1))

    # Scale the initial velocities to control the angular momentum
    angular_velocity_scale = 2000
    velocities *= angular_velocity_scale

    return positions, velocities



def main():
    # Evenly distribute particles in space
    grid_size = int(np.ceil(np.sqrt(Particles)))
    x = np.linspace(0, 1, grid_size) * 2 - 1
    y = np.linspace(0, 1, grid_size) * 2 - 1

    # Initialize positions and velocities
    positions, velocities = initialize_positions(Particles)

    # Initialize GLFW
    if not glfw.init():
        return
    
    window = glfw.create_window(screenSize, screenSize, "Particle Simulation", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)

    # Main loop
    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)
        glfw.make_context_current(window)

        # Draw particles
        draw_particles(positions.cpu().numpy())  # Convert positions tensor to NumPy array

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

        positions, velocities = simulate(positions, velocities, particle_properties, C, dt)
        
        # Check for NaN or Inf values
        if torch.isnan(positions).any() or torch.isinf(positions).any() or \
        torch.isnan(velocities).any() or torch.isinf(velocities).any():
            print("Positions:")
            print(positions)
            print("Velocities:")
            print(velocities)
            break

    glfw.terminate()


if __name__ == "__main__":
    main()
