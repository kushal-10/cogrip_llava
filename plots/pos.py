import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata 

def clean_pos(s):
    if '-' in s:
        return False
    s = s.replace('[ ', '[').replace(' ]', ']')
    s = s.replace('  ', ' ')
    s = s.replace(' ', ',')
    return s

def get_pos(s):
    s = clean_pos(s)
    if not s:
        return False
    if s[2] == ',':
        pos1 = int(s[1])
        if s[4] == ']':
            pos2 = int(s[3])
        elif s[5] == ']':
            pos2 = int(s[3:5])
    else:
        pos1 = int(s[1] + s[2])
        if s[5] == ']':
            pos2 = int(s[4])
        elif s[6] == ']':
            pos2 = int(s[4] + s[5])

    return [pos1, pos2]

def plot_by_pos(csv_path):
    """
    Plot by shape and location
    """
    position_dict = {}
    target_dict = {}
    for i in range(1, 19):
        for j in range(1, 19):
            position_dict[(i, j)] = 0
            target_dict[(i, j)] = 0

    df = pd.read_csv(csv_path)
    success_count = 0
    time_taken = 0
    steps = 0
    pinpoint_success = 0
    for i in range(len(df)):
        time_taken += df.iloc[i]['time']
        steps += df.iloc[i]['steps']
        target_shape = df.iloc[i]['shape']
        # Convert the string representation of the position to a list
        position = df.iloc[i]['target_position']
        predicted_position = df.iloc[i]['predicted_position']
        position = get_pos(position)
        predicted_position = get_pos(predicted_position)
        target_dict[tuple(position)] += 1
        
        if not predicted_position:
            continue

        valid_positions = [position]
        if predicted_position in valid_positions and df.iloc[i]['last_move'] == 'grip':
            pinpoint_success += 1

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

        if predicted_position in valid_positions and df.iloc[i]['last_move'] == 'grip':
            success_count += 1
            position_dict[tuple(predicted_position)] += 1
    
    # After processing the data, plot the heatmap
    # Create a grid for the heatmap
    heatmap_data = np.zeros((18, 18)) 
    redact_data = np.zeros((18, 18))

    success_data = np.zeros((18, 18))
    fail_data = np.zeros((18, 18))

    for i in range(18):
        for j in range(18):
            if target_dict[(i+1, j+1)]:
                    success_data[i, j] = position_dict[(i+1, j+1)]
                    fail_data[i, j] = target_dict[(i+1, j+1)] - position_dict[(i+1, j+1)]
    
    # New code to create continuous heatmaps using interpolation
    x = np.arange(1, 19)
    y = np.arange(1, 19)
    X, Y = np.meshgrid(x, y)

    # Interpolating success and fail data
    success_continuous = griddata((X.flatten(), Y.flatten()), success_data.flatten(), (X, Y), method='cubic')
    fail_continuous = griddata((X.flatten(), Y.flatten()), fail_data.flatten(), (X, Y), method='cubic')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create subplots

    # First continuous heatmap
    axs[0].imshow(success_continuous, cmap='cool', interpolation='bilinear', extent=(1, 18, 1, 18))
    axs[0].set_title('Success cases (Continuous)')
    
    # Second continuous heatmap
    axs[1].imshow(fail_continuous, cmap='inferno', interpolation='bilinear', extent=(1, 18, 1, 18))  
    axs[1].set_title('Failed Cases (Continuous)')

    plt.tight_layout()  # Adjust layout
    plt.show()

if __name__ == "__main__":
    dacc_folder = "realtime_results"
    dacc_csvs = [os.path.join(dacc_folder, 'llava-1.5-7b-hf-ft.csv'), os.path.join(dacc_folder, 'llava-1.5-7b-hf-ft.csv')]
    for csv_path in dacc_csvs:
        plot_by_pos(csv_path)
        break
