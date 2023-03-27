import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

screenSize = 1500

def draw_cube():
    vertices = [
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
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

def main():
    # Initialize Pygame
    pygame.init()
    window = pygame.display.set_mode((screenSize, screenSize), DOUBLEBUF | OPENGL)

    # Set up OpenGL
    gluPerspective(60, 1, 0.1, 100.0)
    glTranslatef(0.0, 0.0, -4)
    glClearColor(0.2, 0.2, 0.2, 1)  # Set a dark gray background color

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
