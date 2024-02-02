import sys
 
# setting path
sys.path.append('../swarmalators')

import swarmalators.swarmalator as sw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import colorsys

agent_count = 5

positions = np.random.uniform(low=-0.5, high=0.5, size=(agent_count, 2))

# positions[:, 1] = 0

# positions = np.array([[-3.85416667e-01,  6.04166667e-02],
#                 [-2.60416667e-01, -3.83333333e-01],
#                 [-1.70833333e-01, -4.79166667e-02],
#                 [ 1.08333333e-01, -1.43750000e-01],
#                 [-1.45833333e-02, -4.79166667e-02],
#                 [-2.08333333e-01, -2.12500000e-01],
#                 [-3.45833333e-01, -9.58333333e-02],
#                 [ 2.29166667e-01, -2.52083333e-01],
#                 [-6.45833333e-02, -3.97916667e-01],
#                 [-4.20833333e-01, -2.45833333e-01],
#                 [ 7.29166667e-02, -3.00000000e-01],
#                 [-5.20833333e-02, -1.95833333e-01],
#                 [-2.08333333e-03,  2.14583333e-01],
#                 [-1.00000000e-01,  8.75000000e-02],
#                 [ 1.33333333e-01,  1.66666667e-02]])

swarm = sw.Swarmalator(agent_count, 1, 1)
# swarm.update(positions[:, :2])

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
    global now
    fig, ax = plt.subplots()

    def update(frame):
        global positions
        global now
        # # Update the model
        swarm.update(positions)
        # # Update position based on previous velocity
        positions += swarm.get_velocity() * (time.time() - now)
        # # positions = np.clip(positions, -1, 1)

        # # print(swarm.get_velocity() * (time.time() - now) * 10)

        # # print(positions)

        phase_state = swarm.get_phase_state()
        colors = angles_to_rgb(phase_state[:, 1]) / 255.0

        # # # Plot the positions
        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], c=colors)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        now = time.time()

    # Set up the animation
    now = time.time()
    animation = FuncAnimation(fig, update, frames=200, interval=10, blit=False)
    plt.show()

now = time.time()
plot_swarm()