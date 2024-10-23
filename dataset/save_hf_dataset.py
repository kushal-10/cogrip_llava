import os
from datasets import load_from_disk
from huggingface_hub import HfApi, HfFolder

# Load the dataset from disk
def load_local_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)
    return dataset

# Upload the dataset to Hugging Face Hub
def upload_to_hub(dataset, repo_name, token):
    api = HfApi()
    # Create a new repo on the Hugging Face Hub
    api.create_repo(repo_name, token=token)
    
    # Save the dataset to a temporary directory
    temp_dir = "temp_dataset"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the dataset files
    dataset.save_to_disk(temp_dir)
    
    # Upload the dataset to the hub
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_name,
        token=token
    )

if __name__ == "__main__":
    
    dataset_path = "training_data/hf_dataset_easy"
    
    # Specify your Hugging Face Hub repo name and token
    repo_name = "Koshti10/llava-pentomino-easy-2"
    token = HfFolder.get_token() 
    
    # Load and upload the dataset
    dataset = load_local_dataset(dataset_path)
    upload_to_hub(dataset, repo_name, token)

