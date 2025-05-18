from datasets import load_dataset
from PIL import Image
import os
import shutil
import json

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("remyxai/OpenSpaces_MC")

# create a root folder for this dataset
dataset_dir = "data/OpenSpaces_MC"

for split in ["train", "test"]:
    sample_dir = os.path.join(dataset_dir, split, "images")
    os.makedirs(sample_dir, exist_ok=True)

    instruct_json = []

    # save images and build initial list entries
    for idx, image in enumerate(ds[split]["images"]):
        image[0].save(os.path.join(sample_dir, f"{idx}.jpg"))
        instruct_json.append({"image": f"{idx}.jpg"})

    # process messages and attach conversations to each entry
    for idx, messages in enumerate(ds[split]["messages"]):
        # extract contents
        z = [x["content"][0] for x in messages]
        z[0] = messages[0]["content"][1]
        # split into question/answer pairs
        texts = [q["text"].split(" Answer: ") for q in z]
        texts = [t for pair in texts for t in pair]
        # format as human/gpt exchanges
        convs = [
            {"from": "human" if i % 2 == 0 else "gpt", "value": text}
            for i, text in enumerate(texts)
        ]
        # attach to the corresponding list entry (idx−1)
        instruct_json[idx]["conversations"] = convs
    
    # save the instruct_json as .json
    with open(os.path.join(dataset_dir, split, "instruct.json"), "w") as f:
        json.dump(instruct_json, f, indent=4)

# # Print the dataset
# print(ds)
# # Print the dataset splits
# print(ds.keys())
# # Print a specific question/answer pair for image 0
# print(ds['train']['messages'][0][2])
# # Show there is image and messages
# print(ds['train'])
# # Print number of Q/A pairs for image 0
# print(len(ds['train']['messages'][0]))
# # Display first image
# Image.open(ds['train']['images'][0]).show()