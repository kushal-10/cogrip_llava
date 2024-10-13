import optuna   
from transformers import AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from huggingface_hub import HfApi

from train_utils.lightning_cls import LlavaModelPLModule

# Set Parameters #
LEVEL='sample'
MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
WANDB_PROJECT = "individual-module"
WANDB_NAME = f"{LEVEL}_tune_1"

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


"""
CONFIG
"""
print("Setting Up CONFIG.............")

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

def objective(trial):  # Define the objective function for Optuna
    # Suggest hyperparameters
    max_epochs = trial.suggest_int("max_epochs", 3, 10)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    
    # Add hyperparameters for LoRA
    r = trial.suggest_int("r", 4, 16)  # Suggest a range for r
    lora_alpha = trial.suggest_int("lora_alpha", 4, 16)  # Suggest a range for lora_alpha
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)  # Suggest a range for lora_dropout

     # Load the model here to ensure it's defined before use
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)

    config = {
        "max_epochs": max_epochs,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 4,
        "lr": lr,
        "batch_size": batch_size,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
        "lora_config": {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
    }

    # Prepare the model with the updated lora_config
    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    # Now you can use the model variable
    model = get_peft_model(model, lora_config)

    model_module = LlavaModelPLModule(config, processor, model)

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
        callbacks=[early_stop_callback],
    )

    trainer.fit(model_module)

    # Return a metric to optimize (e.g., validation loss)
    return trainer.callback_metrics["val_loss"].item()  # Adjust based on your metrics

# Run the hyperparameter tuning
study = optuna.create_study(direction="minimize")  # Create a study
study.optimize(objective, n_trials=10)  # Optimize the objective function

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)
