
from huggingface_hub import hf_hub_download
import os
import zipfile

# Define a main directory
main_dir = "pretrain_data"

# Define subdirectories
sub_dir1 = "chat"
sub_dir2 = "images"

# Create main directory
os.makedirs(main_dir, exist_ok=True)

# Create subdirectories
os.makedirs(os.path.join(main_dir, sub_dir1), exist_ok=True)
os.makedirs(os.path.join(main_dir, sub_dir2), exist_ok=True)

# Define the repository id
repo_id = "liuhaotian/LLaVA-CC3M-Pretrain-595K"

# Download chat.json
hf_hub_download(repo_id=repo_id, filename="chat.json", cache_dir=os.path.join(main_dir, sub_dir1))

# Download images.zip
hf_hub_download(repo_id=repo_id, filename="images.zip", cache_dir=os.path.join(main_dir, sub_dir2))

# Unzip the images.zip
with zipfile.ZipFile(os.path.join(main_dir, sub_dir2, 'images.zip'), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(main_dir, sub_dir2))

print("Files downloaded, moved and images unzipped successfully!")