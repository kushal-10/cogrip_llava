"""
Evaluation strategies

1) Episodic -> Check How many episodes are completed -> Trained on single turn evaluated as single turn.
 - episodic_success_rate out of 540 episodes (3600 episodes total, 2600 in training)
 
2) Raw -> Overall accuracy
 - Out of all instances - 2000 instances
"""

import pandas as pd
import os
import math
import json

with open(os.path.join('additional_jsons', 'shape_map.json'), 'r') as f:
    shape_data = json.load(f)

# csv_path = os.path.join("results", "gpt.csv")

def eval_model(csv_path):

    df = pd.read_csv(csv_path)
    data = {}

    for i in range(len(df)):
        board_number = df.iloc[i]['ids'].replace('"', '').split('/')[3].replace('boards', '')
        if board_number not in data:
            data[board_number] = {}
            data[board_number]['predictions'] = []
            data[board_number]['gts'] = []
        
        data[board_number]['predictions'].append(df.iloc[i]['predictions'])
        data[board_number]['gts'].append(df.iloc[i]['gts'])


    boards = data.keys()

    board_count = 0
    instance_count = 0
    step_accuracy = 0
    precise_success = 0
    success = 0

    for b in boards:
        board_count += 1

        preds = data[b]['predictions']
        gts = data[b]['gts']

        # Initialize position
        position = [0, 0]  # Initial position (x, y)

        # Update position based on ground truth values
        for move in gts:
            instance_count += 1
            move = move.lower()
            if move == 'right':
                position[0] += 1
            elif move == 'left':
                position[0] -= 1
            elif move == 'up':
                position[1] += 1
            elif move == 'down':
                position[1] -= 1
            elif move == 'grip':
                pass  # No change in position for grip

        # Check if move is made in the right direction
        predicted_position = [0, 0]
        backup_preds = preds
        preds = preds[:-1] # Last one should always be grip
        for prediction in preds:
            prediction = prediction.lower()

            prev_position = predicted_position
            prev_distance = math.sqrt((prev_position[0]-position[0])**2 + (prev_position[1]-position[1])**2 )
            if prediction == 'right':
                predicted_position[0] += 1
            elif prediction == 'left':
                predicted_position[0] -= 1
            elif prediction == 'up':
                predicted_position[1] += 1
            elif prediction == 'down':
                predicted_position[1] -= 1

            new_position = predicted_position
            new_distance = math.sqrt((new_position[0]-position[0])**2 + (new_position[1]-position[1])**2 )

            if new_distance < prev_distance: # move was made in the right direction
                step_accuracy += 1

        if predicted_position == position:
            if backup_preds[-1].lower() == 'grip':
                precise_success += 1

        target_shape = shape_data[b]
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

        # valid_positions = set(valid_positions)
        if predicted_position in valid_positions:
            if backup_preds[-1].lower() == 'grip':
                success += 1


    precision_success_rate = precise_success/board_count
    success_rate = success/board_count
    direction_accuracy = step_accuracy/instance_count

    # model_name = csv_path.split('/')[-1].split('.csv')[0]
    # print(f"For model - {model_name} \n\n Success Rate - {success_rate} \n\n Direction Accuracy - {direction_accuracy} \n\n OCD Sucess Rate - {precision_success_rate}")

    return direction_accuracy, success_rate, precision_success_rate

if __name__ == '__main__':    

    model_names = []
    daccs = []
    srs = []
    psrs = []
    result_csvs = ["llava-1.5-7b-hf.csv", "llava-1.5-7b-hf-ft.csv", "llava-1.5-13b-hf.csv", "llava-1.5-13b-hf-ft.csv", "gpt.csv"]
    for pth in result_csvs:
        model_names.append(pth.split('.csv')[0])
        dacc, sr, psr = eval_model(os.path.join("results", pth))
        daccs.append(dacc)
        srs.append(sr)
        psrs.append(psr)

    result_data = {
        'Model': model_names,
        'Direction Accuracy': daccs,
        'Success Rate': srs,
        'Precision Success Rate': psrs
    }

    df = pd.DataFrame(result_data)
    df.to_csv(os.path.join('results', 'result.csv'), index=False)
    
