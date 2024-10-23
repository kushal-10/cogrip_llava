from PIL import Image, ImageFilter
from grip_env.environment import GridWorldEnv
import os
import json
from tqdm import tqdm
import argparse 
import numpy as np
import pandas as pd
import time
import base64
from io import BytesIO
import requests

# LEVEL = 'easy'
# BOARD_SIZE = 18

api_key = os.getenv("OPENAI_API_KEY")

if not os.path.exists('results'):
    os.makedirs('results')

class GPTEval():

    def __init__(self, level: str, board_size: int, max_moves: int, max_length: int):
        self.level = level
        self.board_size = board_size
        self.max_moves = max_moves
        self.max_len = max_length

        if level == 'easy':
            metadata_path = os.path.join('data', level, 'test.json')
        else:
            metadata_path = os.path.join('data', level, 'train.json')

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        new_metadata = []
        for i in range(len(self.metadata)):
            metadata_obj = self.metadata[i]
            for j in range(len(metadata_obj)):
                if "info" in metadata_obj[j]:
                    new_metadata.append(metadata_obj[j])
        self.metadata = new_metadata

    
    @staticmethod
    def convert_move_to_step(move_str):
        if move_str == 'up':
            step_val = 3
        elif move_str == 'down':
            step_val = 1
        elif move_str == 'left':
            step_val = 2
        elif move_str == 'right':
            step_val = 0
        else:
            step_val = 4 # Do nothing/Grip

        return step_val
    
    @staticmethod
    def get_next_position(prediction, predicted_position):
        if prediction == 'right':
            predicted_position[0] += 1
        elif prediction == 'left':
            predicted_position[0] -= 1
        elif prediction == 'up':
            predicted_position[1] -= 1
        elif prediction == 'down':
            predicted_position[1] += 1

        return predicted_position
    
    @staticmethod
    def get_gpt_response(image, base_prompt, max_new_tokens):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": base_prompt + " Answer in one word only."
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}"
                }
                }
            ]
            }
        ],
        "max_tokens": max_new_tokens
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response:
            generated_text = response.json()['choices'][0]['message']['content']
            output = generated_text.lower()
        else:
            print("No response")
        
        return output

    
    def evaluate(self):

        final_moves = []
        final_positions = []
        target_positions = []
        target_shapes = []
        target_colours = []
        target_regions = []
        steps = []
        time_taken = []
        total_steps_taken = []
        
        for i in tqdm(range(90)):
            metadata_obj = self.metadata[i]
            info = metadata_obj['info']
            agent_start_pos = np.array(metadata_obj['agent_start_pos'])
            target_pos = np.array(metadata_obj['target_pos'])

            target_shape = metadata_obj['target_shape']
            target_color = metadata_obj['target_color']
            target_region = info[0]['piece_region']

            base_prompt = f"You are at the black dot in the board. The target is the {target_color} {target_shape} piece located at the {target_region}. Your task is to move towards the target and grab it. Predict your next move from up, down, left, right, grip."
            
            env = GridWorldEnv(render_mode="rgb_array", size=self.board_size, grid_info=info, agent_pos=agent_start_pos, target_pos=target_pos)
            env.reset()
            image = env.render()
            image = Image.fromarray(image)

            predicted_position = agent_start_pos
            total_time = 0  
            total_steps = 0  
            start_time = time.time() 
            steps_taken = []
            for j in range(self.max_moves):
                total_steps += 1
                
                move = self.get_gpt_response(image, base_prompt, self.max_len)
                env.step(self.convert_move_to_step(move))
                image = env.render()
                image = Image.fromarray(image)
                steps_taken.append(move)

                predicted_position = self.get_next_position(move, predicted_position)
                final_move = move
                final_position = predicted_position

                if move == 'grip':
                    break
            
            total_time = time.time() - start_time

            final_moves.append(final_move)
            final_positions.append(final_position)
            target_positions.append(target_pos)
            target_colours.append(target_color)
            target_regions.append(target_region)
            target_shapes.append(target_shape)
            steps.append(total_steps)
            time_taken.append(total_time)
            total_steps_taken.append(steps_taken)
        
        model_save_name = 'gpt'
        prediciton_data = {
            'last_move': final_moves,
            'predicted_position': final_positions,
            'target_position': target_positions,
            'shape': target_shapes,
            'region': target_regions,
            'color': target_colours,
            'steps': steps,
            'time': time_taken,
            'total_steps_taken': total_steps_taken
        }

        pred_df = pd.DataFrame(prediciton_data)
        pred_df.to_csv(os.path.join('results', f'{model_save_name}_{self.level}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaVA Evaluation Parameters')
    parser.add_argument('--level', type=str, default='easy', help='Level of the game')
    parser.add_argument('--board_size', type=int, default=18, help='Size of the board')
    parser.add_argument('--max_moves', type=int, default=20, help='Maximum number of moves')
    parser.add_argument('--max_length', type=int, default=5, help='Maximum length of the output')

    args = parser.parse_args()  # Parse the arguments

    eval = GPTEval(args.level, args.board_size, args.max_moves, args.max_length)
    eval.evaluate()





    