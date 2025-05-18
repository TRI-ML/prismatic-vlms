from pathlib import Path
from scripts.pretrain import pretrain, PretrainConfig
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

    cfg = PretrainConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        stage="finetune",
        run_id="vggt_projector_run",
        # run_root_dir=Path("runs"),
        seed=42,
        hf_token="",  # Adjust if your token is elsewhere
        trackers=("jsonl",),
        # wandb_project="onyx-vlms",
        # wandb_entity="stanford-voltron",
        pretrained_checkpoint="/home/scur0690/.cache/huggingface/hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/dinov2-224px+7b/checkpoints/latest-checkpoint.pt",
    )
    pretrain(cfg)
