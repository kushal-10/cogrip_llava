from PIL import Image, ImageFilter
from grip_env.environment import GridWorldEnv
import os
import json
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
import torch
import argparse  # Add this import

# LEVEL = 'easy'
# BOARD_SIZE = 18

class LLaVAEval():

    def __init__(self, level: str, board_size: int, model_name: str, max_moves: int, max_length: int):
        self.level = level
        self.board_size = board_size
        self.model_name = model_name
        self.max_moves = max_moves
        self.max_len = max_length

        metadata_path = os.path.join('realtime_eval', f'metadata_{self.level}.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def load_model_and_processor(self):
        MODEL_ID = self.model_name

        if "-ft" in MODEL_ID:
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            # Define quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            # Load the base model with adapters on top
            model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )
        else:
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
            model.to("cuda")

        return model, processor
    
    def evaluate(self, model, processor):
        MODEL_ID = self.model_name
        for i in tqdm(range(len(self.metadata))):
            metadata_obj = self.metadata[i]
            info = metadata_obj['info']
            agent_start_pos = metadata_obj['agent_start_pos']
            target_pos = metadata_obj['target_pos']

            target_shape = metadata_obj['target_shape']
            target_color = metadata_obj['target_color']
            target_region = info[0]['piece_region']

            base_prompt = f"You are at the black dot in the board. The target is the {target_color} {target_shape} piece located at the {target_region}. Your task is to move towards the target and grab it. Predict your next move from up, down, left, right, grip."
            if "-ft" in MODEL_ID:
                prompt = f"USER: <image>\n{base_prompt}.\nASSISTANT:"
            else:
                prompt = f"USER: <image>\n{base_prompt}. Answer in one word only.\nASSISTANT:"

            env = GridWorldEnv(render_mode="rgb_array", size=self.board_size, grid_info=info, agent_pos=agent_start_pos, target_pos=target_pos)
            env.reset()
            image = env.render()
            image = Image.fromarray(image)


            inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_len)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            # Extract move
            move = generated_texts[0].split('ASSISTANT:')[-1].strip()
            move = move.lower()

            print(move)

            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaVA Evaluation Parameters')
    parser.add_argument('--level', type=str, default='easy', help='Level of the game')
    parser.add_argument('--board_size', type=int, default=18, help='Size of the board')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--max_moves', type=int, default=10, help='Maximum number of moves')
    parser.add_argument('--max_length', type=int, default=10, help='Maximum length of the output')

    args = parser.parse_args()  # Parse the arguments

    eval = LLaVAEval(args.level, args.board_size, args.model_name, args.max_moves, args.max_length)
    model, processor = eval.load_model_and_processor()
    eval.evaluate(model, processor)





    