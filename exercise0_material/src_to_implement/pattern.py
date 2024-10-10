import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output =  None
        if resolution % (2 * tile_size) != 0:
            raise Exception("Resolution must be a multiple of 2 * tile_size")

    def draw(self):
        # Hint: Use np.hstack and np.vstack to create the pattern
        # Hint: Use np.vstack to stack the rows of the pattern
        # Hint: Use np.tile to repeat the pattern
        # Hint: Use np.zeros to create the output array
        black_tile = np.ones((self.tile_size, self.tile_size), dtype=int)
        white_tile = np.zeros((self.tile_size, self.tile_size), dtype=int)
        # black white
        # white black
        element_tile = np.vstack((np.hstack((white_tile,black_tile)),np.hstack((black_tile,white_tile))))
        # Create the output array by tiling the element_tile
        self.output = np.tile(element_tile, (self.resolution // (2 * self.tile_size), self.resolution // (2 * self.tile_size)))

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Circle:

    def __init__(self, resolution: int, radius: int,position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        # Create a grid of coordinates
        xv, yv = np.meshgrid(np.arange(0, self.resolution), np.arange(0, self.resolution))
        # Calculate the distance of each point from the center
        distance_from_center = np.sqrt((xv - self.position[0]) ** 2 + (yv - self.position[1]) ** 2)
        # Generate a circular mask, True for points within or on the circle
        # if use "[distance_from_center <= self.radius]" will get (1,resolution,resolution) shape array. Error
        self.output = (distance_from_center <= self.radius).astype(int)
        # Return a copy of the output to prevent external modifications
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution,3))

        # Set red channel: linear gradient from left to right (0 to 1)
        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)
        # Set green channel: linear gradient from top to bottom (0 to 1)
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        # Set blue channel: linear gradient from left to right (1 to 0)
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()

