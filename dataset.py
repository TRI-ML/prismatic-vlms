from datasets import load_dataset
from PIL import Image
import os
import shutil
import json

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("remyxai/OpenSpaces_MC")

# create a root folder for this dataset
dataset_dir = "OpenSpaces_MC"

for split in ["train", "test"]:
    for idx, (image, messages) in enumerate(zip(ds[split]["images"], ds[split]["messages"]), start=1):
        sample_dir = os.path.join(dataset_dir, split, str(idx))
        os.makedirs(sample_dir, exist_ok=True)

        image[0].save(os.path.join(sample_dir, f"{idx}.jpg"))

        # save messages as .json
        # For some reason the first message is not in the format of the rest
        z = [x['content'][0] for x in messages]
        z[0] = messages[0]['content'][1]
        # split the text into question and answer
        texts = [q['text'].split(" Answer: ") for q in z]
        # flatten the nested lists in texts
        texts = [t for sublist in texts for t in sublist]
        # Add in the format that prismatic vlm wants
        messages = [
            {"from": "human" if i % 2 == 0 else "gpt", "value": text} for i, text in enumerate(texts)
        ]
        # save the image
        msg_out = os.path.join(sample_dir, f"{idx}.json")
        with open(msg_out, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

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