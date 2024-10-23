import os
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

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


fail_data = np.zeros((18,18))

def plot_fails(csv_path):
    df = pd.read_csv(csv_path)
    for i in range(len(df)):
        position = df.iloc[i]['target_position']
        predicted_position = df.iloc[i]['predicted_position']
        position = get_pos(position)
        predicted_position = get_pos(predicted_position)

    
        if not predicted_position:
            # Invalid position predicted, skip 
            continue

        valid_positions = [position]
        if not (predicted_position in valid_positions and df.iloc[i]['last_move'] == 'grip'):
            fail_data[position[0], position[1]] += 1

    plt.figure(figsize=(9, 9))
    plt.imshow(fail_data, cmap='cool', interpolation='bilinear', extent=(1, 18, 18, 1))  # Create the heatmap
    plt.colorbar()  # Add a colorbar to indicate the scale
    plt.title('Heatmap showing the positions of target pieces which the model failed to grip')  # Add a title
    plt.xlabel('X Position')  # Label for x-axis
    plt.ylabel('Y Position')  # Label for y-axis
    plt.savefig(f'plots/{csv_path.split("/")[-1].split(".csv")[0]}_fails.png', bbox_inches='tight')  # Save the plot locally
    plt.close()





if __name__ == "__main__":
    dacc_folder = "results"
    dacc_csvs = ["results/llava-1.5-13b-hf-ft-2ep_easy.csv"]
    for csv_path in dacc_csvs:
        plot_fails(csv_path)
