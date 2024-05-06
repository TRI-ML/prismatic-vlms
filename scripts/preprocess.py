"""
preprocess.py

Core script for automatically downloading raw VLM pretraining datasets. Supports downloading the following datasets:
    - LLaVA v1.5 Datasets (for both training stages) [`llava-laion-cc-sbu-558k`, `llava-v1.5-instruct`]
        - Stage 1 :: Projection Matrix Alignment between Vision Encoder & Pretrained LLM on CC-3M-595K (Custom)
        - Stage 2 :: Projection & LLM Finetuning on LLaVa v1.5 Instruct (including various vision-language train sets)

By default, runs download & extraction automatically.

Run with: `python scripts/preprocess.py --dataset_id <DATASET_ID>`
"""

from dataclasses import dataclass
from pathlib import Path

import draccus

from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import convert_to_jpg, download_extract

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PreprocessConfig:
    # fmt: off
    dataset_id: str = "llava-v1.5-instruct"                     # Unique identifier for dataset to process (see above)
    root_dir: Path = Path("data")                               # Path to root directory for storing datasets

    # fmt: on


@draccus.wrap()
def preprocess(cfg: PreprocessConfig) -> None:
    overwatch.info(f"Downloading & Extracting `{cfg.dataset_id}` to `{cfg.root_dir / 'download'}")
    download_extract(cfg.dataset_id, root_dir=cfg.root_dir)

    # Special Handling for OCR VQA Images (for `llava-v1.5-instruct`) --> convert GIFs/PNGs to JPG
    if cfg.dataset_id == "llava-v1.5-instruct":
        convert_to_jpg(cfg.root_dir / "download" / cfg.dataset_id / "ocr_vqa" / "images")


if __name__ == "__main__":
    preprocess()
