from datasets import load_dataset
import os
import zipfile

# Load the dataset
dataset = load_dataset('liuhaotian/LLaVA-CC3M-Pretrain-595K')

# Define a main directory
main_dir = "main_directory"

# Define subdirectories
sub_dir1 = "chat"
sub_dir2 = "images"

# Create main directory
os.makedirs(main_dir, exist_ok=True)

# Create subdirectories
os.makedirs(os.path.join(main_dir, sub_dir1), exist_ok=True)
os.makedirs(os.path.join(main_dir, sub_dir2), exist_ok=True)

# Save chat.json
with open(os.path.join(main_dir, sub_dir1, 'chat.json'), 'w') as f:
    json.dump(dataset['train'], f)

# Save images.zip
with open(os.path.join(main_dir, sub_dir2, 'images.zip'), 'wb') as f:
    f.write(dataset['train']['images'])

# Unzip the images.zip
with zipfile.ZipFile(os.path.join(main_dir, sub_dir2, 'images.zip'), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(main_dir, sub_dir2))