import os
import json
import random
from tqdm import tqdm

metadata_path = os.path.join('data', 'sample', 'metadata.json')
save_path = os.path.join('data', 'sample', 'metadata_path.json')

# Generate random shortest path
def generate_random_path(start, end):
    x1, y1 = start
    x2, y2 = end
    
    # Calculate the number of moves in each direction
    horizontal_moves = x2 - x1
    vertical_moves = y2 - y1
    
    # Create move lists
    moves = []
    if horizontal_moves > 0:
        moves += ['right'] * horizontal_moves
    elif horizontal_moves < 0:
        moves += ['left'] * -horizontal_moves
    
    if vertical_moves > 0:
        moves += ['down'] * vertical_moves  # Y increases down
    elif vertical_moves < 0:
        moves += ['up'] * -vertical_moves
    
    # Shuffle the moves for randomness
    random.shuffle(moves)
    return moves

with open(metadata_path, 'r') as f:
    data = json.load(f)


paths_data = []    
for i in tqdm(range(len(data)), desc="Generating paths for the dataset"):
    data_obj = data[i]
    point1 = list(data[i]['agent_start_pos'])
    point2 = list(data[i]['target_pos'])

    path = generate_random_path(point1, point2)
    
    data_obj['path'] = path

    paths_data.append(data_obj)

with open(save_path, 'w') as f:
    json.dump(paths_data, f, indent=4)






