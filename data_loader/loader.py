import os
import numpy as np
import pandas as pd
import json
from datasets import Dataset, Features, Image, Value, Sequence


def generate_hf_parquet_dataset(entries):
    features = Features(
        {
            "actions": Sequence(Value("int32")),
            "scores": Sequence(Value("int32")),
            "images": Sequence(Image())
        }
    )
    dataset = Dataset.from_list(entries, features=features)

    return dataset


def format_data(images_file, eps_data_file):
    # load images
    images = np.load(images_file)

    # load actions and scores array
    with open(eps_data_file, 'r') as f:
        eps_data = json.load(f)

    actions = eps_data["action"]
    scores = eps_data["score"]

    # convert into lists
    actions = [int(x[0]) for x in actions]
    scores = [int(x[0]) for x in scores]
    images = [x for x in images]

    return actions, scores, images


entries = []

parent_dir = "example_dataset"

for dir in os.listdir(parent_dir):
    # try to find smb-1-1-xxx
    child_dir = os.path.join(parent_dir, dir)
    
    # if valid directory
    if dir.startswith("smb-1-1") and os.path.isdir(child_dir):
        print("parsing through: " + child_dir)
        current_dir_completed = set()
        
        # loop through files in directory
        for file in os.listdir(child_dir):
            try:
                # get file number, ensure not previously formatted
                file_number = file.split("data_")[1].split(".")[0]
                if file_number not in current_dir_completed:
                    # get data files
                    images_file = os.path.join(child_dir, f"screen_data_{file_number}.npy")
                    print("formatting data files: " + images_file)
                    eps_data_file = os.path.join(child_dir, f"eps_data_{file_number}.json")
                    print("formatting data files: " + eps_data_file)
                    
                    # format data
                    actions, scores, images = format_data(images_file, eps_data_file)
                    entries.append({
                        "actions": actions,
                        "scores": scores,
                        "images": images
                    })
                    current_dir_completed.add(file_number)
            except Exception as e:
                print("file invalid: " + child_dir + "\\" + file + ". Exception: " + e)

# unused
# df = pd.DataFrame({
#     'scores': [scores],
#     'actions': [actions],
#     'images': [images]
# })

dataset = generate_hf_parquet_dataset(entries)

# save locally
dataset.save_to_disk(f"{parent_dir}/smb_rl_dataset")

# save to your Hugging Face dataset
# dataset.push_to_hub("Gabaloo1/SMB_RL")