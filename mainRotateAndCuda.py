import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Additional imports
import numpy as np
import torch
import cv2
import pygame.surfarray as surfarray


screenSize = 1500

# Constants
NumberDimensions = 3
Particles = 8000
N = 2
c = 100000.0  # Set the speed of light or another relativistic speed limit
damping = 0.4

print(torch.cuda.is_available())
def get_device():
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

device = get_device()

# Coupling matrix
C = torch.tensor([[5000., 0.],
                  [0., 0.]]).to(device)

# Particle properties
particle_properties = torch.ones(Particles, N).to(device)


# Add repulsive force at the walls
wall_force_strength = -500
wall_force_thickness = 0.1

central_force_strength = -3000.0
    
# Simulation loop
dt = 5e-7

def draw_axes(camera_position):
    glBegin(GL_LINES)

    # Define axes endpoints
    axes_endpoints = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0]
    ])

    # Define colors for the axes
    colors = [
        [1.0, 0.0, 0.0],  # x-axis - red
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],  # y-axis - green
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],  # z-axis - blue
        [0.0, 0.0, 1.0]
    ]

    for i, endpoint in enumerate(axes_endpoints):
        x, y, z = endpoint
        glColor3f(*colors[i])
        glVertex3f(x, y, z)

    glEnd()

def draw_cube():
    vertices = [
        [0.6, -0.6, -1.8],
        [0.6, 0.6, -1.8],
        [-0.6, 0.6, -1.8],
        [-0.6, -0.6, -1.8],
        [0.6, -0.6, -0.8],
        [0.6, 0.6, -0.8],
        [-0.6, -0.6, -0.8],
        [-0.6, 0.6, -0.8],
    ]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_particles(positions):
    positions_cpu = positions.cpu().numpy()
    N = positions.shape[0]

    # Camera position
    camera_position = np.array([0, 0, -1])

    # glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_POINTS)
    for pos in positions_cpu:
        x, y, z = pos
        glColor3f(1.0, 1.0, 0.0)  # Use a more visible color
        glVertex3f(x, y, z)
    glEnd()

    draw_axes(camera_position)

    
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
    distances = torch.norm(displacement_vectors, dim=2, keepdim=True).to(device)
    distances += torch.eye(Particles)[:, :, None].to(device)  # Add identity to avoid division by zero

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
    relativistic_factor = (1 - (velocities ** 2).sum(dim=-1) / c ** 2 + epsilon) ** (-1/2)


    accelerations = total_forces / particle_properties[:, 0:1] * relativistic_factor.unsqueeze(-1) * damping

    # Update positions and velocities
    positions += velocities * dt + 0.5 * accelerations * dt**2
    velocities += accelerations * dt

    return positions, velocities


import numpy as np

def initialize_positions(Particles):
    # Generate random angles
    angles = torch.rand((Particles, 1), dtype=torch.float32, device=device) * 2 * np.pi
    phi = torch.rand((Particles, 1), dtype=torch.float32, device=device) * np.pi

    # Generate random radii from a normal distribution
    radii = torch.randn((Particles, 1), dtype=torch.float32, device=device) * 0.5

    # Compute the x, y, and z coordinates from the angles and radii
    x = radii**2 * torch.sin(phi) * torch.cos(angles)
    y = radii**2 * torch.sin(phi) * torch.sin(angles)
    z = radii**2 * torch.cos(phi)

    # Combine x, y, and z coordinates
    positions = torch.stack((x, y, z), dim=-1).squeeze()

    # Calculate the initial velocities
    axis = torch.tensor([0.0, 1.0, 0.], device=device)
    velocities = torch.cross(positions, axis.repeat(Particles, 1))

    # Scale the initial velocities to control the angular momentum
    angular_velocity_scale = 10000
    velocities *= angular_velocity_scale

    return positions, velocities

def set_perspective(fov_y, aspect, z_near, z_far):
    f_h = np.tan(fov_y / 360.0 * np.pi) * z_near
    f_w = f_h * aspect
    glFrustum(-f_w, f_w, -f_h, f_h, z_near, z_far)



def main():
    device = get_device()

    # Initialize Pygame
    pygame.init()
    window = pygame.display.set_mode((screenSize, screenSize), DOUBLEBUF | OPENGL)

    clock = pygame.time.Clock()  # Add this line to create a clock

    # Set up OpenGL
    set_perspective(60, 1, 0.1, 100.0)
    glDisable(GL_CULL_FACE)
    glTranslatef(0.0, 0.0, -4)

    # Initialize positions and velocities
    positions, velocities = initialize_positions(Particles)

    # Mouse and rotation state
    mouse_down = False
    rotation_x, rotation_y = 0, 0

    # Camera distance
    fov = 90
    min_fov = 20
    max_fov = 120
    
    output_video = 'output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30  # Adjust the FPS to match the desired video speed
    frame_size = (screenSize, screenSize)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size, isColor=True)


    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    prev_x, prev_y = event.pos
                # Zoom in
                elif event.button == 4:
                    fov = max(fov - 3, min_fov)
                # Zoom out
                elif event.button == 5:
                    fov = min(fov + 3, max_fov)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down:
                    x, y = event.pos
                    dx, dy = x - prev_x, y - prev_y
                    rotation_x += dy * 0.5
                    rotation_y += dx * 0.5
                    prev_x, prev_y = x, y

        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        set_perspective(fov, 1, 0.1, 100.0)
        glDisable(GL_CULL_FACE)
        glTranslatef(0.0, 0.0, -4)
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        draw_particles(positions)

        pygame.display.flip()
        
        # Capture and write the frame to the video file
        buffer = glReadPixels(0, 0, screenSize, screenSize, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(screenSize, screenSize, 3)
        frame_bgr = cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

        positions, velocities = simulate(positions, velocities, particle_properties, C, dt)

        # Check for NaN or Inf values
        if torch.isnan(positions).any() or torch.isinf(positions).any() or \
           torch.isnan(velocities).any() or torch.isinf(velocities).any():
            print("Positions:")
            print(positions)
            print("Velocities:")
            print(velocities)
            break
        
    # Release the video writer
    video_writer.release()
    pygame.quit()

if __name__ == "__main__":
    main()
    
    
