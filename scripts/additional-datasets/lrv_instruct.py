"""
scripts/additional-datasets/lrv_instruct.py

Standalone script for pre-processing the LRV-Instruct data (including the chart/diagram reasoning split). This isn't
full conversational chat data, but rather each example has an input prompt and output response; we'll use this structure
to format the data equivalently to the LLaVa-v1.5 dataset.

In general, LRV Instruct provides *both positive and negative* examples -- where a negative example is a question or
instruction that is *not answerable* or *irrelevant*; the goal of this dataset is to reduce hallucinations in VLMs.

This script downloads the raw instruct data (three different JSON files), as well as the image files; the non-chart
images come from Visual Genome, but are hosted separately by the LRV Instruct authors and use different image IDs, so
we're downloading this data (again) for simplicity. The chart images come from the LRV Instruct authors, and are sourced
from statista.com. All file URLS are here: https://github.com/FuxiaoLiu/LRV-Instruction/blob/main/download.txt#L20

Note that we are using the *coordinate-free* data (due to noted inaccuracies in the original coordinates).

Make sure to download the images first to `data/download/llava-v1.5-instruct/lrv`
    => cd data/download/llava-v1.5-instruct/lrv
    => [Visual Genome] gdown https://drive.google.com/uc?id=1k9MNV-ImEV9BYEOeLEIb4uGEUZjd3QbM
        => `tar -xvf image.tar.gz; mv image lrv-vg; rm image.tar.gz`
    => [Chart Data] gdown https://drive.google.com/uc?id=1Dey-undzW2Nl21CYLFSkP_Y4RrfRJkYd
        => `unzip chart_image.zip; rm -rf __MACOSX; mv chart_image lrv-chart; rm chart_image.zip`

Download the raw JSON files to the same directory - `data/download/llava-v1.5-instruct/lrv`
    => [LRV Instruct Pt. 1] gdown https://drive.google.com/uc?id=1pWkxE2kqpys1VdwBi99ZXN6-XY5SqhwU
        => `filter_cap1.json`
    => [LRV Instruct Pt. II] gdown https://drive.google.com/uc?id=1NTxkuRPlvDn7aWaJpK_yb0p5r0cxPLNZ
        => `filter_cap_more1.json`
    => [Chart Instruct] gdown https://drive.google.com/uc?id=13j2U-ectsYGR92r6J5hPdhT8T5ezItHF
        => `chart_release_update.json`

References: "Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning"
    => Paper: https://arxiv.org/abs/2306.14565
    => Github / Data: https://github.com/FuxiaoLiu/LRV-Instruction
"""

import json
import random
from pathlib import Path

from tqdm import tqdm

# === Constants ===
BASE_DIR = Path("data/download/llava-v1.5-instruct")
LRV_DIR = BASE_DIR / "lrv"

VG_JSON_FILES, VG_IMG_DIR = [LRV_DIR / "filter_cap1.json", LRV_DIR / "filter_cap_more1.json"], LRV_DIR / "lrv-vg"
CHART_JSON_FILE, CHART_IMG_DIR = LRV_DIR / "chart_release_update.json", LRV_DIR / "lrv-chart"

# JSON Files for "merged" variants fo the dataset (with `llava_v1_5_mix665k.json` and `llava_v1_5_lvis4v_mix888k.json`
BASE_JSON_FILE = BASE_DIR / "llava_v1_5_mix665k.json"
BASE_LVIS_JSON_FILE = BASE_DIR / "llava_v1_5_lvis4v_mix888k.json"

MERGED_BASE_LRV_JSON_FILE = BASE_DIR / "llava_v1_5_lrv_mix1008k.json"
MERGED_BASE_LVIS_LRV_JSON_FILE = BASE_DIR / "llava_v1_5_lvis4v_lrv_mix1231k.json"


def build_lrv_instruct() -> None:
    print("[*] Downloading and Formatting `LRV-Instruct` Dataset!")

    # Set Random Seed
    random.seed(7)

    # Open VG JSON Files
    vg_examples = []
    for fn in VG_JSON_FILES:
        with open(fn, "r") as f:
            vg_examples.extend(json.load(f))

    # Iterate through VG Examples & Verify Image Existence
    for example in tqdm(vg_examples, desc="[*] Verifying all VG Images in LRV Instruct"):
        image_id = example["image_id"]
        assert (VG_IMG_DIR / f"{image_id}.jpg").exists(), f"Missing Image `{image_id}.jpg`"

    # Open Chart JSON File
    with open(CHART_JSON_FILE, "r") as f:
        chart_examples = json.load(f)

    # Iterate through Chart Examples & Verify Image Existence
    for example in tqdm(chart_examples, desc="[*] Verifying all Chart Images in LRV Instruct"):
        image_path = example["image_id"]
        assert (CHART_IMG_DIR / image_path).exists(), f"Missing Image `{image_path}`"

    # Reformat VG Examples as LLaVa "Chat" Style => List[Entry] where each Entry is a Dictionary:
    #   => "id": str
    #   => "image": str -- Relative path from `BASE_DIR`
    #   => "conversations: List[Turn] where Turn is a Dictionary:
    #           => {"from": "human", "value": "<image>\n{VG_EXAMPLE['question']}"}
    #           => {"from": "gpt", "value": "{VG_EXAMPLE['answer']}"}
    vg_chat_json = []
    for vg_example in tqdm(vg_examples, desc="[*] Converting all VG Examples to LLaVa Format"):
        vg_chat_json.append(
            {
                "id": vg_example["image_id"],
                "image": f"lrv/lrv-vg/{vg_example['image_id']}.jpg",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{vg_example['question'].strip()}"},
                    {"from": "gpt", "value": vg_example["answer"].strip()},
                ],
            }
        )

    # Reformat Chart Examples as LLaVa "Chat" Style
    chart_chat_json = []
    for chart_example in tqdm(chart_examples, desc="[*] Converting all Chart Examples to LLaVa Format"):
        chart_chat_json.append(
            {
                "id": Path(chart_example["image_id"]).stem,
                "image": f"lrv/lrv-chart/{chart_example['image_id']}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{chart_example['question'].strip()}"},
                    {"from": "gpt", "value": chart_example["answer"].strip()},
                ],
            }
        )

    # Merge and Create Full LRV Chat Data =>> Total of 342,799 Examples
    lrv_data = vg_chat_json + chart_chat_json

    # Create Stacked Datasets =>> Shuffle for Good Measure!
    print("[*] Loading LLaVa v1.5 Data!")
    with open(BASE_JSON_FILE, "r") as f:
        llava_v15_data = json.load(f)

    # Combine & Shuffle & Write
    llava_lrv_data = llava_v15_data + lrv_data

    random.shuffle(llava_lrv_data)
    random.shuffle(llava_lrv_data)
    random.shuffle(llava_lrv_data)

    with open(MERGED_BASE_LRV_JSON_FILE, "w") as f:
        json.dump(llava_lrv_data, f)

    print("[*] Loading LLaVa v1.5 + LVIS-4V Instruct Data!")
    with open(BASE_LVIS_JSON_FILE, "r") as f:
        llava_v15_lvis_data = json.load(f)

    # Combine & Shuffle & Write
    full_data = llava_v15_lvis_data + lrv_data

    random.shuffle(full_data)
    random.shuffle(full_data)
    random.shuffle(full_data)

    with open(MERGED_BASE_LVIS_LRV_JSON_FILE, "w") as f:
        json.dump(full_data, f)


if __name__ == "__main__":
    build_lrv_instruct()
