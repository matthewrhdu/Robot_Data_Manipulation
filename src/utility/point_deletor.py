from __future__ import annotations
import pygame
import numpy as np
import math
from typing import Optional, List

filename = "acc.npy"  # CHANGE ME!!!!

# Setup the screen
background_colour = (0, 0, 0)
width, height = 1800, 1000
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Simulation')

screen.fill(background_colour)

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


start_x, start_y = width // 2, 6 * height // 8
spacing = 500


def coordinates(x_pos: float = 0, y_pos: float = 0):
    return start_x + x_pos * spacing, start_y - y_pos * spacing


def rev_coordinates(x_pos: float = 0, y_pos: float = 0):
    return (x_pos - start_x) / spacing, (start_y - y_pos) / spacing


class Box:
    def __init__(self):
        self.start = None
        self.curr = None
        self.end = None

    def display(self):
        if self.curr is not None:
            sides = [
                (self.start[0], self.start[1]),
                (self.start[0], self.curr[1]),
                (self.curr[0], self.curr[1]),
                (self.curr[0], self.start[1])
            ]

            pygame.draw.polygon(screen, GREEN, sides, 5)


# A particle
class Particle:
    def __init__(self, x, y, size_, raw_point):
        self.x, self.y = coordinates(x, y)
        self.size = size_
        self.colour = WHITE
        self.thickness = 0
        self.raw_point = raw_point

    def display(self):
        pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.size, self.thickness)

    def __str__(self):
        return f"({self.x}, {self.y})"


# Main code
def main():
    pts = np.load(filename)
    matrix = np.array(
        [
            np.array([1, -0.5 ** 0.5, 0]),
            np.array([0, -0.5 ** 0.5, 1]),
            np.array([0, 0, 0])
        ]
    )

    pts_t = np.array([np.matmul(matrix, pt) for pt in pts])
    my_particles = [Particle(pts_t[pt_id][0], pts_t[pt_id][1], 3, pts[pt_id]) for pt_id in range(len(pts_t))]

    box = Box()
    starting = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not starting:
                    starting = True
                    box.start = pygame.mouse.get_pos()
                elif starting:
                    box.end = pygame.mouse.get_pos()
                    starting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    box.start = None
                    box.curr = None
                    box.end = None
                    starting = False

        if starting:
            box.curr = pygame.mouse.get_pos()

        if box.start and box.end:
            max_x, min_x = max([box.start[0], box.end[0]]), min([box.start[0], box.end[0]])
            max_y, min_y = max([box.start[1], box.end[1]]), min([box.start[1], box.end[1]])

            new = []
            for particle in my_particles:
                if not (min_x <= particle.x <= max_x and min_y <= particle.y <= max_y):
                    new.append(particle)

            my_particles = new
            box.start = None
            box.curr = None
            box.end = None

        screen.fill(background_colour)

        for particle in my_particles:
            particle.display()
        box.display()

        pygame.display.flip()

    remaining = np.array([pt.raw_point for pt in my_particles])
    np.save(f"{filename[:-len('.npy')]}_cleaned.npy", remaining)


if __name__ == '__main__':
    main()
