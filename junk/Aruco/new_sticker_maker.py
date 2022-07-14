from __future__ import annotations
import pygame
import numpy as np

SIZE = 30
input_filename = "MBL_Logo.npy"
save_filename = input_filename

# Set up the screen
background_colour = (255, 255, 255)
width, height = 900, 900
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Sticker!')

screen.fill(background_colour)

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

start_x, start_y = width // 2, height // 2


def find_particle(particles, x, y):
    for p in particles:
        if p.x_range[0] <= x <= p.x_range[1] and p.y_range[0] <= y <= p.y_range[1]:
            return p


class Particle:
    def __init__(self, center: tuple, spacing: int):
        self.center_x = center[0]
        self.center_y = center[1]
        self.selected = True
        self.spacing = spacing
        self.x_range = (self.center_x - spacing // 2, self.center_x + spacing // 2)
        self.y_range = (self.center_y - spacing // 2, self.center_y + spacing // 2)
        self.lock = False

    def display(self):
        sides = [
            (self.center_x + self.spacing // 2, self.center_y + self.spacing // 2),
            (self.center_x + self.spacing // 2, self.center_y - self.spacing // 2),
            (self.center_x - self.spacing // 2, self.center_y - self.spacing // 2),
            (self.center_x - self.spacing // 2, self.center_y + self.spacing // 2)
        ]
        if self.selected:
            pygame.draw.polygon(screen, BLACK, sides, 0)
        else:
            pygame.draw.polygon(screen, BLACK, sides, 5)


# Main code
def main():
    spacing = width // SIZE
    start = spacing // 2
    my_particles = [Particle((start + i * spacing, start + j * spacing), spacing) for i in range(SIZE) for j in range(SIZE)]

    preset_data(my_particles)

    selected_particle = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        mouse_keys = pygame.mouse.get_pressed()
        if mouse_keys[0]:  # mouse down
            mouseX, mouseY = pygame.mouse.get_pos()
            selected_particle = find_particle(my_particles, mouseX, mouseY)
        elif selected_particle is not None:  # release mouse
            for part in my_particles:
                part.lock = False
            selected_particle = None

        if selected_particle is not None and not selected_particle.lock:
            selected_particle.selected = not selected_particle.selected
            selected_particle.lock = True
            # selected_particle = None

        screen.fill(background_colour)

        for particle in my_particles:
            particle.display()

        pygame.display.flip()

    arr = np.empty((SIZE, SIZE), dtype=np.uint8)
    x_pos = 0
    y_pos = 0
    for part in my_particles:
        if not part.selected:
            arr[x_pos][y_pos] = 1
        else:
            arr[x_pos][y_pos] = 0

        if x_pos >= SIZE - 1:
            x_pos = 0
            y_pos += 1
        else:
            x_pos += 1

    np.save(save_filename, arr)


def preset_data(my_particles):
    read = np.load(input_filename)
    n = 0
    for i in range(read.shape[0]):
        for j in range(read.shape[1]):
            if not read[j][i] == 1:
                my_particles[n].selected = False
            n += 1


if __name__ == "__main__":
    main()
