"""
Calls GridWorldEnv to create instances of an episode with different agent positions using Gym
Save details to a json file under traindata/level/*boards_i/*step1.png....
"""

import json
from tqdm import tqdm
import os
import numpy as np
import argparse
from PIL import Image, ImageFilter

from grip_env.environment import GridWorldEnv
from grip_env.layout import BoardLayout
from grip_env.pieces import PIECE_NAMES, COLOUR_NAMES

# Additonal handler class for numpy arrays
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)
    

def episodes(level='sample', board_size=15):
    """
    Generate and save instances of episodes for a given level
    args: 
        :level: The level for which the instances needs to be generated 
    """

    image_save_dir = os.path.join("training_data", level, "boards")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    training_metadata_path = os.path.join("training_data", level, "training_metadata.json")

    # Load the metadata with paths information
    metadata_path = os.path.join("data", level, "metadata_path.json")
    with open(metadata_path, 'r') as f:
        paths_data = json.load(f)

    # A step map to convert the step tokens to gym compatible actions
    step_map = {
        "right": 0,
        "down": 1,
        "left": 2,
        "up": 3
    }

    metadata = []

    for i in tqdm(range(len(paths_data)), desc="Generating instances for {level}"):
        board_dir = os.path.join(image_save_dir, f'boards{i}')
        if not os.path.exists(board_dir):
            os.makedirs(board_dir)

        data_obj = paths_data[i]
        episode_obj = []

        info = data_obj['info']
        agent_start_pos = np.array(data_obj['agent_start_pos'])
        target_pos = np.array(data_obj['target_pos'])
        paths = data_obj['path']

        target_shape = data_obj['target_shape']
        target_color = data_obj['target_color']
        target_region = info[0]['piece_region']
        prompt = f"You are at the black dot in the board. The target is the {target_color} {target_shape} piece located at the {target_region}. Your task is to move towards the target and grab it. Predict your next move from up, down, left, right, grip."
        instance_object = {}


        env = GridWorldEnv(render_mode="rgb_array", size=board_size, grid_info=info, agent_pos=agent_start_pos, target_pos=target_pos)
        env.reset()
        image = env.render()
        image = Image.fromarray(image)  # Convert to PIL Image if using numpy array
        image_save_path = os.path.join(board_dir, f'step_{0}.png')
        image.save(image_save_path)  # Save the image
        instance_object['image'] = image_save_path
        instance_object['prompt'] = prompt
        

        for k in range(len(paths)):
            step = paths[k]
            # The next step, is the output for prev response
            instance_object['ground_truth'] = step
            episode_obj.append(instance_object)
            instance_object = {} # Reset instance obj after saving prev one

            action = step_map[step]
            env.step(action)

            image = env.render()
            image = Image.fromarray(image) 
            image_save_path = os.path.join(board_dir, f'step_{k+1}.png')
            image.save(image_save_path)
            instance_object['image'] = image_save_path
            instance_object['prompt'] = prompt
        
        
        instance_object['ground_truth'] = 'grip'
        episode_obj.append(instance_object)
        episode_obj.append(data_obj)

        metadata.append(episode_obj)
    
    # Save the meta data with image locations, prompt and response for each instance
    with open(training_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Generated and saved instances for Level: {level}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=15, help='Number of grids NxN - multiples of 3 are better')
    parser.add_argument('--level', type=str, default='sample', help='Difficulty level - String - "easy", "medium", "hard"')
    args = parser.parse_args()

    episodes(args.level, args.size)

"""
python3 dataset/generate_episodes.py --size 15 --level 'sample'
"""
