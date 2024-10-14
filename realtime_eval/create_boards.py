"""
Calls BoardLayout and GridWorldEnv to create a board layout and render it
Save details to a json file under data/difficulty_metadata.json
"""

import json
from tqdm import tqdm
import os
import numpy as np
import argparse

from grip_env.layout import BoardLayout
from grip_env.pieces import PIECE_NAMES, COLOUR_NAMES

# Additonal handler class for numpy arrays
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)

def boards(board_size, num_pieces, shapes, colours, num_boards, level, pth):
    
    metadata_list = []  # List to store metadata for each board
    for shape in PIECE_NAMES:
        for colour in COLOUR_NAMES:
            for i in tqdm(range(num_boards), desc=f"Generating boards for {shape} {colour}"):
            
                seed = np.random.randint(0, 100000)
                board = BoardLayout(board_size=board_size, 
                                    num_pieces=num_pieces, 
                                    shapes=shapes, 
                                    colours=colours, 
                                    seed=seed)
                
                agent_start_pos, target_pos, info = board.set_board_layout(target_shape=shape,
                                                                           target_colour=colour,
                                                                           level=level) 
    
                # # Use render_mode="rgb_array" to get a numpy array of the board and save the image
                # env = GridWorldEnv(render_mode="rgb_array", size=board_size, grid_info=info, agent_pos=agent_start_pos, target_pos=target_pos)
                # env.reset()
                # image = env.render()

                # Collect metadata
                metadata = {
                    'agent_start_pos': agent_start_pos,
                    'target_pos': target_pos,
                    'target_shape': shape,
                    'target_color': colour,
                    'info': info
                }
                metadata_list.append(metadata)  # Append metadata to the list
                

    # Save metadata to JSON file
    with open(os.path.join(pth, f'metadata_{level}.json'), 'w') as json_file:
        json.dump(metadata_list, json_file, cls=NumpyArrayEncoder, indent=4)  # Write metadata to JSON file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', type=int, default=18, help='Size of the board - Integer')
    parser.add_argument('--num_pieces', type=int, default=4, help='Number of pieces - Integer - 2, 4, 8, 16')
    parser.add_argument('--shapes', type=str, default=' '.join(PIECE_NAMES), help='Shapes of the pieces - Space-separated string')
    parser.add_argument('--colours', type=str, default=' '.join(COLOUR_NAMES), help='Colours of the pieces - Space-separated string')
    parser.add_argument('--num_boards', type=int, default=100, help='Number of boards to create for each combination of piece and colour - Integer - Total number of boards is shapes * colours * num_boards')
    parser.add_argument('--level', type=str, default='easy', help='Difficulty level - String - "easy", "medium", "hard"')
    parser.add_argument('--path', type=str, default='data', help='Output file to save boards - String')
    
    args = parser.parse_args()
    args.shapes = args.shapes.split()  # Convert to list
    args.colours = args.colours.split()  # Convert to list

    boards(args.board_size, args.num_pieces, args.shapes, args.colours, args.num_boards, args.level, args.path)
