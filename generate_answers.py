from pathlib import Path
from scripts.generate import generate, GenerateConfig
from prismatic.conf import DatasetRegistry, ModelRegistry, ModelConfig, DatasetConfig
import torch
import torch.distributed as dist

if __name__ == "__main__":
    # Set up config to match train_vggt.py
    model_id = "dinov2-224px+7b"
    dataset_id = DatasetRegistry.OPENSPACES.dataset_id
    model_cfg = ModelConfig.get_choice_class(model_id)
    dataset_cfg = DatasetConfig.get_choice_class(dataset_id)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    cfg = GenerateConfig(model_path=model_cfg)

    generate(cfg)