import os
import json
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata 

with open("additional_jsons/easy/train.json", "r") as f:
    train_data = json.load(f)
with open("additional_jsons/easy/val.json", "r") as f:
    val_data = json.load(f)
with open("additional_jsons/easy/test.json", "r") as f:
    test_data = json.load(f)



def get_valid_positions(target_shape, position):
    valid_positions = [position]
    if target_shape == 'P':
        valid_positions.append([position[0]+1, position[1]])
        valid_positions.append([position[0]+1, position[1]+1])
        valid_positions.append([position[0], position[1]+1])
        valid_positions.append([position[0], position[1]-1])
    elif target_shape == 'T':
        valid_positions.append([position[0]-1, position[1]+1])
        valid_positions.append([position[0]+1, position[1]+1])
        valid_positions.append([position[0], position[1]+1])
        valid_positions.append([position[0], position[1]-1])
    elif target_shape == 'U':
        valid_positions.append([position[0]-1, position[1]+1])
        valid_positions.append([position[0]+1, position[1]+1])
        valid_positions.append([position[0]+1, position[1]])
        valid_positions.append([position[0]-1, position[1]])
    elif target_shape == 'W':
        valid_positions.append([position[0]-1, position[1]+1])
        valid_positions.append([position[0]-1, position[1]])
        valid_positions.append([position[0], position[1]-1])
        valid_positions.append([position[0]+1, position[1]-1])
    elif target_shape == 'X':
        valid_positions.append([position[0]-1, position[1]])
        valid_positions.append([position[0]+1, position[1]])
        valid_positions.append([position[0], position[1]+1])
        valid_positions.append([position[0], position[1]-1])
    elif target_shape == 'Z':
        valid_positions.append([position[0]-1, position[1]+1])
        valid_positions.append([position[0], position[1]+1])
        valid_positions.append([position[0], position[1]-1])
        valid_positions.append([position[0]+1, position[1]-1])

    return valid_positions

target_positions = np.zeros((18, 18))
for i in range(len(train_data)):
    episode_data = train_data[i]
    position = [9,9]
    shape = episode_data[0]["prompt"].split(" ")[14]

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
    valid_positions = get_valid_positions(shape, position)
    for valid_position in valid_positions:
        target_positions[valid_position[0], valid_position[1]] += 1

for i in range(len(val_data)):
    episode_data = val_data[i]
    position = [9,9]
    shape = episode_data[0]["prompt"].split(" ")[14]
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
    valid_positions = get_valid_positions(shape, position)
    for valid_position in valid_positions:
        target_positions[valid_position[0], valid_position[1]] += 1

test_positions = np.zeros((18, 18))
for i in range(len(test_data)):
    episode_data = test_data[i]
    position = [9,9]
    shape = episode_data[0]["prompt"].split(" ")[14]

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
    test_positions[position[0], position[1]] += 1
    valid_positions = get_valid_positions(shape, position)
    for valid_position in valid_positions:
        test_positions[valid_position[0], valid_position[1]] += 1

x = np.arange(1, 19)
y = np.arange(1, 19)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(9, 9))
plt.imshow(target_positions, cmap='gist_rainbow_r', interpolation='bilinear', extent=(1, 18, 18, 1))  # Create the heatmap
plt.colorbar()  # Add a colorbar to indicate the scale
plt.title('Target Pieces Layout Heatmap - Train + Val Split')  # Add a title
plt.xlabel('X Position')  # Label for x-axis
plt.ylabel('Y Position')  # Label for y-axis
plt.savefig('plots/target_positions_heatmap_training.png', bbox_inches='tight')  # Save the plot locally


plt.figure(figsize=(9, 9))
plt.imshow(test_positions, cmap='gist_rainbow_r', interpolation='bilinear', extent=(1, 18, 18, 1))  # Create the heatmap
plt.colorbar()  # Add a colorbar to indicate the scale
plt.title('Target Pieces Layout Heatmap - Test Split')  # Add a title
plt.xlabel('X Position')  # Label for x-axis
plt.ylabel('Y Position')  # Label for y-axis
plt.savefig('plots/target_positions_heatmap_test.png', bbox_inches='tight')  # Save the plot locally


