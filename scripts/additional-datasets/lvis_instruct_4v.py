"""
scripts/additional-datasets/lvis_instruct4v.py

Standalone script for pre-processing the LVIS-Instruct4V (language/chat) data (`lvis_instruct4v_220k.json`). This
dataset is curated from LVIS images (subset of COCO yet again), but chat data is synthesized from GPT4-Vision.

This script downloads the raw data, merges with the LLaVa v15 data, and performs any other data normalization, saving
the resulting `.json` file(s) to the `data/download/llava-v1.5-instruct/` directory.

Make sure to download the COCO Val 2017 (LVIS) data to `data/download/llava-v1.5-instruct/coco`:
    => cd data/download/llava-v1.5-instruct/coco
    => wget http://images.cocodataset.org/zips/val2017.zip
    => unzip val2017.zip; rm val2017.zip

References: "To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning"
    => Paper: https://arxiv.org/abs/2311.07574
    => Github / Data: https://github.com/X2FD/LVIS-INSTRUCT4V || https://huggingface.co/datasets/X2FD/LVIS-Instruct4V
"""

import json
import os
import random
from pathlib import Path

from tqdm import tqdm

from prismatic.preprocessing.download import download_with_progress

# === Constants ===
DATA_URL = "https://huggingface.co/datasets/X2FD/LVIS-Instruct4V/resolve/main/lvis_instruct4v_220k.json"
DOWNLOAD_DIR = Path("data/download/llava-v1.5-instruct")
RAW_JSON_FILE = DOWNLOAD_DIR / "lvis_instruct4v_220k.json"

# JSON Files for "merged" variant of the dataset (with `llava_v1_5_mix665k.json`)
BASE_JSON_FILE = DOWNLOAD_DIR / "llava_v1_5_mix665k.json"
MERGED_JSON_FILE = DOWNLOAD_DIR / "llava_v1_5_lvis4v_mix888k.json"


def build_lvis_instruct_4v() -> None:
    print("[*] Downloading and Formatting `LVIS-Instruct-4V` Dataset!")

    # Set Random Seed
    random.seed(7)

    # Download Dataset JSON
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    if not RAW_JSON_FILE.exists():
        download_with_progress(DATA_URL, DOWNLOAD_DIR)

    # Open JSON File --> verify image existence!
    print("[*] Loading LVIS Instruct4V Data!")
    with open(RAW_JSON_FILE, "r") as f:
        data = json.load(f)

    # Iterate & Verify
    for example in tqdm(data, desc="[*] Verifying all Images in LVIS Instruct4V"):
        image_path = example["image"]
        assert (DOWNLOAD_DIR / image_path).exists(), f"Missing Image `{image_path}`"

    # Create Stacked Dataset =>> Shuffle for Good Measure!
    print("[*] Loading LLaVa v1.5 Data!")
    with open(BASE_JSON_FILE, "r") as f:
        llava_v15_data = json.load(f)

    # Combine & Shuffle & Write
    full_data = llava_v15_data + data

    random.shuffle(full_data)
    random.shuffle(full_data)
    random.shuffle(full_data)

    with open(MERGED_JSON_FILE, "w") as f:
        json.dump(full_data, f)


if __name__ == "__main__":
    build_lvis_instruct_4v()
