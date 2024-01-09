import sys
 
# setting path
sys.path.append('../swarmalators')

import swarmalators.swarmalator as sw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import colorsys

positions = np.random.uniform(low=-1.0, high=1.0, size=(100, 2))
# positions = np.array([[0.25, 0], [0.0, 0.0]])

swarm = sw.Swarmalator(100, 0, 1)

# swarm.update(positions)

# print(swarm.get_velocity())

def angles_to_rgb(angles_rad):
    # Convert the angles to hue values (ranges from 0.0 to 1.0 in the HSV color space)
    hues = angles_rad / (2 * np.pi)

    # Set fixed values for saturation and value (you can adjust these as desired)
    saturation = np.ones_like(hues)
    value = np.ones_like(hues)

    hsv_colors = np.stack((hues, saturation, value), axis=-1)
    rgb_colors = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv_colors)

    # Scale RGB values to 0-255 range
    rgb_colors *= 255
    rgb_colors = rgb_colors.astype(np.uint8)

    return rgb_colors

def plot_swarm():
    fig, ax = plt.subplots()

    def update(frame):
        global positions
        global now
        # Update the model
        swarm.update(positions)
        # Update position based on previous velocity
        positions += swarm.get_velocity() * (time.time() - now)
        # positions = np.clip(positions, -1, 1)

        print(positions)

        phase_state = swarm.get_phase_state()
        colors = angles_to_rgb(phase_state[:, 1]) / 255.0

        # # Plot the positions
        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], c=colors)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        now = time.time()

    # Set up the animation
    animation = FuncAnimation(fig, update, frames=200, interval=10, blit=False)
    plt.show()

now = time.time()
plot_swarm()