import os
import json
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata 

with open("additional_jsons/easy/train.json", "r") as f:
    train_data = json.load(f)
with open("additional_jsons/easy/val.json", "r") as f:
    val_data = json.load(f)


target_positions = np.zeros((18, 18))
for i in range(len(train_data)):
    episode_data = train_data[i]
    position = [9,9]
    for j in range(len(episode_data)):
        move = episode_data[j]["ground_truth"]
        if move == 'right':
            position[0] += 1
        elif move == 'left':
            position[0] -= 1
        elif move == 'up':
            position[1] -= 1
        elif move == 'down':
            position[1] += 1
    target_positions[position[0], position[1]] += 1

for i in range(len(val_data)):
    episode_data = val_data[i]
    position = [9,9]
    for j in range(len(episode_data)):
        move = episode_data[j]["ground_truth"]
        if move == 'right':
            position[0] += 1
        elif move == 'left':
            position[0] -= 1
        elif move == 'up':
            position[1] -= 1
        elif move == 'down':
            position[1] += 1
    target_positions[position[0], position[1]] += 1

x = np.arange(1, 19)
y = np.arange(1, 19)
X, Y = np.meshgrid(x, y)

target_continuous = griddata((X.flatten(), Y.flatten()), target_positions.flatten(), (X, Y), method='cubic')
    

# After processing target_positions, add the following code to plot the heatmap
plt.imshow(target_continuous, cmap='hot', interpolation='bilinear', extent=(1, 18, 1, 18))  # Create the heatmap
plt.colorbar()  # Add a colorbar to indicate the scale
plt.title('Target Positions Heatmap')  # Add a title
plt.xlabel('X Position')  # Label for x-axis
plt.ylabel('Y Position')  # Label for y-axis
plt.show()  # Display the heatmap




