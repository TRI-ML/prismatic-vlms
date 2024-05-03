# Prismatic VLMs

[![arXiv](https://img.shields.io/badge/arXiv-2402.07865-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2402.07865)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Usage**](#usage) | [**Pretrained Models**](#pretrained-models) | [**Training VLMs**](#training-vlms)

A flexible and efficient codebase for training visually-conditioned language-models (VLMs):

- **Different Visual Representations**. We natively support backbones such as [CLIP](https://arxiv.org/abs/2103.00020), 
  [SigLIP](https://arxiv.org/abs/2303.15343), [DINOv2](https://arxiv.org/abs/2304.07193) – and even fusions of different backbones. 
  Adding new backbones is easy via [TIMM](https://huggingface.co/timm).
- **Base and Instruct-Tuned Language Models**. We support arbitrary instances of `AutoModelForCausalLM` including both 
  base and instruct-tuned models (with built-in prompt handling) via [Transformers](https://github.com/huggingface/transformers). 
  If your favorite LM isn't already supported, feel free to submit a PR!
- **Easy Scaling**. Powered by PyTorch FSDP and Flash-Attention, we can quickly and efficiently train models from 1B - 
  34B parameters, on different, easily configurable dataset mixtures.

If you're interested in rigorously evaluating existing VLMs, check our [evaluation codebase](https://github.com/TRI-ML/vlm-evaluation)
that bundles together 11 different battle-tested vision-and-language benchmarks through a clean, automated test harness. 

---

## Installation

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require
PyTorch 2.1 or greater installation instructions [can be found here](https://pytorch.org/get-started/locally/). This 
repository was developed and has been thoroughly tested with PyTorch 2.1.0, Torchvision 0.16.0, and Flash-Attention 2.3.3.

Once PyTorch has been properly installed, you can install this package locally via an editable installation (or via
`pip install git+https://github.com/TRI-ML/prismatic-vlms`):

```bash
git clone https://github.com/TRI-ML/prismatic-vlms
cd prismatic-vlms
pip install -e .

# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja wheel

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2 
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install flash-attn --no-build-isolation
```

If you run into any problems during the installation process, please file a GitHub Issue.

## Usage

Once installed, loading and running inference with pretrained `prismatic` models is easy:

```python
import requests
import torch

from PIL import Image
from pathlib import Path

from prismatic import load

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "prism-dinosiglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
user_prompt = "What is going on in this image?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=1,
)
```

For a complete terminal-based CLI for interacting with our VLMs, check out [scripts/generate.py](scripts/generate.py). 

## Pretrained Models

We release **all 42** VLMs trained as part of our work, with a range of different visual representations, language
models, data, and scale. The exhaustive set of models (with structured descriptions) can be found in 
[`prismatic/models/registry.py](prismatic/models/registry.py) - we will continue to update this registry as we train
additional models.

We also provide a top-level API for instantiating models from the names mentioned in the various Figures of our paper,
as well as for generally browsing our pretrained models by description:

```python
from prismatic import available_model_names, available_models, get_model_description
from pprint import pprint

# List all Pretrained VLMs (by HF Hub IDs)
pprint(available_models())

# List all Pretrained VLMs + Descriptions (by explicit labels / names from paper figures)
pprint(available_model_names())

# Print and return a targeted description of a model (by name or ID) 
#   =>> See `prismatic/models/registry.py` for explicit schema
description = get_model_description("Prism-DINOSigLIP 13B (Controlled)")
```

Currently, our best performing models are the `Prism-DINOSigLIP` series, with especially strong performance on spatial
understanding and localization tasks.

---
**Explicit Notes on Model Licensing & Commercial Use**: While all code in this repository is released under an MIT
License, our pretrained models may inherit restrictions from the _datasets_ and _underlying LMs_ we use for training. 

**[02/09/24]** Our current VLMs are all derived from Llama-2, and as such are subject to the 
[Llama Community License](https://ai.meta.com/llama/license/), which does permit commercial use. We additionally train 
on the LLaVa Instruct Tuning data.

As we train new models, we will update this section of the README (and the LICENSE files associated with each model)
appropriately. If there are any questions, please file an Issue!

## Training VLMs

In addition to providing all pretrained VLMs trained in this work, we also provide full instructions and configurations
for _reproducing all results_ (down to controlling for the batch order of examples seen during training). 

#### Pretraining Datasets
For the [LLaVa v1.5 Instruct Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) we use for all 
of our models, we provide an automated download script in [`scripts/preprocess.py`](scripts/preprocess.py):

```bash
# Download the `llava-v1.5-instruct` (Instruct Tuning) Image and Language Data (includes extra post-processing)
python scripts/preprocess.py --dataset_id "llava-v1.5-instruct" --root_dir <PATH-TO-DATA-ROOT>

# (In case you also wish to download the explicit vision-language alignment data)
python scripts/preprocess.py --dataset_id "llava-laion-cc-sbu-558k" --root_dir <PATH-TO-DATA-ROOT>
```

As part of our work, we also train on mixtures of datasets including 
[LVIS-Instruct-4V](https://arxiv.org/abs/2311.07574) and [LRV-Instruct](https://arxiv.org/abs/2306.14565). We provide
instructions and scripts for downloading these datasets in [`scripts/additional-datasets`](scripts/additional-datasets).

We welcome any and all contributions and pull requests to add new datasets!

#### Model Configuration & Training Script

The entry point for training models is [`scripts/pretrain.py`](scripts/pretrain.py). We employ 
[`draccus`](https://pypi.org/project/draccus/0.6/) to provide a modular, dataclass-based interface for specifying 
model configurations; all 42 VLM configurations are in [`prismatic/conf/models.py`](prismatic/conf/models.py). 

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs, though we also provide a simpler
Distributed Data Parallel training implementation (for smaller LM backbones, debugging). You can run a pretraining job
via `torchrun`.

As a compact example, here's how you would train a VLM derived from Vicuña-v1.5 7B, using fused DINOv2 + SigLIP 
representations, processing non-square images with a "letterbox padding" transform across 8 GPUs on a single-node: 

```bash
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "<NAME OF NEW MODEL>" \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "letterbox" \
  --model.llm_backbone_id "vicuna-v15-7b" 
```

Note that specifying `model.type` is important for identifying the _base configuration_ that you want to build on top of;
the full list of model types are available in our [config file](prismatic/conf/models.py), under the `model_id` key for 
each dataclass.

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `scripts/` - Standalone scripts for preprocessing, training VLMs, and generating from pretrained models.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!

---

#### Citation 

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2402.07865):

```bibtex
@article{karamcheti2024prismatic,
  title = {Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models},
  author = {Siddharth Karamcheti and Suraj Nair and Ashwin Balakrishna and Percy Liang and Thomas Kollar and Dorsa Sadigh},
  journal = {arXiv preprint arXiv:2402.07865},
  year = {2024},
}
```
