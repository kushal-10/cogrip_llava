import os
import pandas as pd
import json
import ast  # Add this import at the top

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

def eval_realtime(csv_path):
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
    
    print(f"Success Rate: {success_count/len(df)}", f"Pinpoint Success Rate: {pinpoint_success/len(df)}", f"Time Taken: {time_taken/len(df)}", f"Steps: {steps/len(df)}", csv_path)
        

def eval_dacc(csv_path):

    df = pd.read_csv(csv_path)
    # Initialize the board map with 0,0 as initial position.
    board_map = {}
    for i in range(len(df)):
        board_num = str(df.iloc[i]['ids'].replace('"', '')).split('/')[-2].replace("boards", "")
        if board_num not in board_map:
            board_map[board_num] = [0,0]

        move = df.iloc[i]['gts']
        # Map to normal cartesian coordinates instead of gym coordinates
        if move == 'right':
            board_map[board_num][0] += 1
        elif move == 'left':
            board_map[board_num][0] -= 1
        elif move == 'up':
            board_map[board_num][1] += 1
        elif move == 'down':
            board_map[board_num][1] -= 1

    initial_positions = {}
    total_moves = 0
    correct_moves = 0
    for i in range(len(df)):
        total_moves += 1
        board_num = str(df.iloc[i]['ids'].replace('"', '')).split('/')[-2].replace("boards", "")
        if board_num not in initial_positions:
            initial_positions[board_num] = [0,0]
        target_position = board_map[board_num]

        initial_distance = (target_position[0] - initial_positions[board_num][0])**2 + (target_position[1] - initial_positions[board_num][1])**2

        predicted_move = df.iloc[i]['predictions']
        if predicted_move == 'right':
            initial_positions[board_num][0] += 1
        elif predicted_move == 'left':
            initial_positions[board_num][0] -= 1
        elif predicted_move == 'up':
            initial_positions[board_num][1] += 1
        elif predicted_move == 'down':
            initial_positions[board_num][1] -= 1

        final_distance = (target_position[0] - initial_positions[board_num][0])**2 + (target_position[1] - initial_positions[board_num][1])**2

        if final_distance < initial_distance:
            correct_moves += 1

    print(f"Directional Accuracy: {correct_moves/total_moves}", csv_path)


if __name__ == "__main__":
    dacc_folder = "results"
    dacc_csvs = [os.path.join(dacc_folder, f) for f in os.listdir(dacc_folder) if f.endswith('.csv')]
    for csv_path in dacc_csvs:
        eval_dacc(csv_path)

    dacc_folder = "realtime_results"
    dacc_csvs = [os.path.join(dacc_folder, f) for f in os.listdir(dacc_folder) if f.endswith('.csv')]
    for csv_path in dacc_csvs:
        eval_realtime(csv_path)
