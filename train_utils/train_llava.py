from transformers import AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch

"""
Refer - https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb
"""

# Set Parameters #
MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REPO_ID = "Koshti10/llava-1.5-7b-ft-sample"
WANDB_PROJECT = "individual-module"
WANDB_NAME = "Finetuned"

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

USE_LORA = False
USE_QLORA = True

"""

Load model

"""
# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning

if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )
    # Skipped FLash Attn for now, no cuda support for local user, needs root access


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
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

