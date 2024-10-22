from transformers import AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForVision2Seq
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from huggingface_hub import HfApi
import json 
import os

from train_utils.lightning_cls import LlavaModelPLModule
from train_utils.llava_next_cls import LlavaNextModelPLModule

"""
Refer - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb
"""

# Load configuration from JSON file
with open(os.path.join('train_utils', 'train_config.json'), 'r') as f:
    train_config = json.load(f)

# Set Parameters for logging
LEVEL = train_config.get("LEVEL")
MODEL_ID = train_config.get("MODEL_ID")
REPO_ID = train_config.get("REPO_ID")
WANDB_PROJECT = train_config.get("WANDB_PROJECT")
WANDB_NAME = train_config.get("WANDB_NAME")

# Lora and Lightning parameters
LORA_R = train_config.get("R")
LORA_ALPHA = train_config.get("ALPHA")
LORA_DROPOUT = train_config.get("DROPOUT")
EPOCHS = train_config.get("EPOCHS")
LR = train_config.get("LR")
BATCH_SIZE = train_config.get("BATCH_SIZE")

torch.set_float32_matmul_precision('high')


processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

USE_LORA = False
USE_QLORA = True

print("Loading the model.................")

"""
Load model
"""

if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

"""
CONFIG
"""
print("Setting Up CONFIG.............")
config = {"max_epochs": EPOCHS,
          "val_check_interval": 0.2,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 2,
          "lr": LR,
          "batch_size": BATCH_SIZE,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

# if 'mistral' or 'vicuna' in MODEL_ID:
#     model_module = LlavaNextModelPLModule(config, processor, model)
# else:
#     model_module = LlavaModelPLModule(config, processor, model)
model_module = LlavaModelPLModule(config, processor, model)

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Push model to the hub every 0.2 epochs
        if trainer.current_epoch % 0.1 == 0:
            print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
            pl_module.model.push_to_hub(REPO_ID,
                                        commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(REPO_ID,
                                        commit_message=f"Training done")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[PushToHubCallback(), early_stop_callback],
)
print("Start Training")
trainer.fit(model_module)
