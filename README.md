# cogrip_llava

## Overview
The `cogrip_llava` repository is designed for the Individual Module.

#### Usage
**Clone the Repository**:
   ```bash
   git clone https://github.com/kushal-10/cogrip_llava.git
   cd cogrip_llava
   ```

**Install Dependencies**:
   Ensure you have the required libraries installed. You can do this using pip:
   ```bash
   pip install -r requirements.txt
   ```

##### 1) Create Boards
The `dataset/create_boards.py` script is responsible for generating data boards from raw dataset inputs. It processes the input data, organizes it into a structured format, and saves the output for further analysis or training purposes.

**Run the Script**:
   Execute the script with the necessary parameters:
   ```bash
   python dataset/create_boards.py --board_size <size> --num_pieces <number> --shapes <shapes> --colours <colours> --num_boards <number> --level <difficulty> --path <output_path>
   ```

**Parameters**:
- `--board_size`: Size of the board (default: 18).
- `--num_pieces`: Number of pieces to use (default: 4).
- `--shapes`: Space-separated string of shapes (default: "P T X Z U W").
- `--colours`: Space-separated string of colors (default: "red blue green yellow magenta cyan").
- `--num_boards`: Total number of boards to create for each combination of piece and color (default: 100).
- `--level`: Difficulty level (default: 'easy').
- `--path`: Output directory to save the generated boards (default: 'data').

**Example**:
   ```bash
   python dataset/create_boards.py --board_size 18 --num_pieces 4 --shapes "P T X Z U W" --colours "red blue green yellow magenta cyan" --num_boards 100 --level 'easy' --path 'data'
   ```

##### 2) Generate Paths from `agent_start_pos` to `target_pos`
Set variables under `dataset/generate_paths`, then run

```bash
python3 dataset/generate_paths.py --level 'easy'
```

This will create a `data/level/metadata_path.json` that saves an additional key of `path` in the `metadata.json` file


##### 3) Now with the paths generated, create the whole training/finetuning dataset that will be used for the llava models

Run the follwing script with additional arguments, if required( level, size) from step1 - Create boards. This will create a folder `training_data/level/boards` and a meta data file that contains `image_path`, `prompt` and `response` for each instance under `training_data/level/training_metadata.json`

```bash
python3 dataset/generate_episodes.py --size 18 --level 'easy'
```

##### 4) Create splits

```bash
python3 dataset/create_hf_dataset.py --level 'easy'
```
This saves `hf_dataset_{level}` under `training_data`. Set the level parameter in the script accordingly


##### 5) Finetuning

```bash
python3 train_utils/train_llava.py
```

Setup training details like model name, HF repo, wandb details etc. in `train_utils/train_config.json` 


##### 6) Real-time Evaluation


```bash
python realtime_eval/evaluate_gpt.py --level easy --board_size 18 --max_moves 20 --max_length 10

python realtime_eval/evaluate_llava.py --level easy --board_size 18 --model_name llava-hf/llava-1.5-7b-hf --max_moves 20 --max_length 10
```

This saves result csv file sunder `inference_results`


#### Paper Evaluation

To generate the results used in the paper, run the following script:

```bash
python3 paper_eval.py
```

