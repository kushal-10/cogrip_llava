import os
import pandas as pd
import ast
import numpy as np
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

def eval(csv_path):
    df = pd.read_csv(csv_path)
    success_count = 0
    time_taken = 0
    steps = 0
    pinpoint_success = 0
    action_success = 0
    total_actions = 0
    for i in range(len(df)):
        time_taken += df.iloc[i]['time']
        steps += df.iloc[i]['steps']
        target_shape = df.iloc[i]['shape']
        position = df.iloc[i]['target_position']
        predicted_position = df.iloc[i]['predicted_position']
        steps_taken = df.iloc[i]['total_steps_taken']
        steps_taken = ast.literal_eval(steps_taken)
        position = get_pos(position)
        predicted_position = get_pos(predicted_position)

        initial_position = [9,9]
        initial_distance = np.linalg.norm(np.array(initial_position) - np.array(position))
        for step in steps_taken:
            if step == 'up':
                initial_position[1] -= 1
            elif step == 'down':
                initial_position[1] += 1
            elif step == 'left':
                initial_position[0] -= 1
            elif step == 'right':
                initial_position[0] += 1

            distance = np.linalg.norm(np.array(initial_position) - np.array(position))
            if distance < initial_distance:
                action_success += 1
            total_actions += 1

        
    
        if not predicted_position:
            # Invalid position predicted, skip 
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
    
    print(f"Success Rate: {success_count/len(df)}", f"Pinpoint Success Rate: {pinpoint_success/len(df)}", f"Action Success Rate: {action_success/total_actions}", f"Time Taken: {time_taken/len(df)}", f"Steps: {steps/len(df)}", csv_path)

if __name__ == "__main__":
    dacc_folder = "results"
    dacc_csvs = [os.path.join(dacc_folder, f) for f in os.listdir(dacc_folder) if f.endswith('.csv')]
    for csv_path in dacc_csvs:
        eval(csv_path)
