"""
convert_prismatic_weights_to_hf.py

Utility script for converting full Prismatic VLM weights (from this repository, in the default "Prismatic" format) to
the HuggingFace "AutoClasses" (e.g., those defined in `prismatic.extern.hf_*`) for "native" use in `transformers``
via `trust_remote_code = True`.

Theoretically, these changes should be fully compatible with directly merging the models into `transformers` down the
line, with first-class support.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import draccus
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.models.vision_transformer import LayerScale
from transformers import AutoTokenizer

from prismatic.extern.hf.configuration_prismatic import PrismaticConfig
from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class HFConvertConfig:
    # fmt: off
    prismatic_model_path_or_id: Union[str, Path] = (                    # Path to Pretrained VLM (on disk or HF Hub)
        "siglip-224px+7b"
        # "prism-dinosiglip-224px+7b"
    )
    output_hf_model_local_path: Path = Path(                            # Path to Local Path to save HF model
        "hf-convert/prismatic-siglip-224px-7b"
    )
    output_hf_model_hub_path: str = (                                   # Path to HF Hub Path for "final" HF model
        "TRI-ML/prismatic-siglip-224px-7b"                              #   => huggingface.co/TRI-ML/prismatic-{...}
    )

    # HF Hub Credentials (required for Gated Models like LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")                      # Environment variable or Path to HF Token

    def __post_init__(self) -> None:
        self.hf_token = self.hf_token.read_text().strip() if isinstance(self.hf_token, Path) else self.hf_token

    # fmt: on


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Conversion Constants ===
PROJECTOR_KEY_MAPPING = {
    "projector.0.weight": "projector.fc1.weight",
    "projector.0.bias": "projector.fc1.bias",
    "projector.2.weight": "projector.fc2.weight",
    "projector.2.bias": "projector.fc2.bias",
    "projector.4.weight": "projector.fc3.weight",
    "projector.4.bias": "projector.fc3.bias",
}


def remap_state_dicts_for_hf(
    projector_state_dict: Dict[str, torch.Tensor],
    llm_backbone_state_dict: Dict[str, torch.Tensor],
    vision_backbone_state_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Iterate through Prismatic component state dictionaries and unify / fix key mapping for HF conversion."""
    hf_state_dict = {}

    # Iterate through Projector =>> use `PROJECTOR_KEY_MAPPING`
    for key, value in projector_state_dict.items():
        hf_state_dict[PROJECTOR_KEY_MAPPING[key]] = value

    # Iterate through LLM Backbone =>> replace `llm.` with `language_model.`
    for key, value in llm_backbone_state_dict.items():
        hf_state_dict[key.replace("llm.", "language_model.")] = value

    # Iterate through Vision Backbone =>> add "vision_backbone." prefix
    assert len(vision_backbone_state_dicts) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
    for idx, vision_backbone_state_dict in enumerate(vision_backbone_state_dicts):
        prefix = "vision_backbone.featurizer" if idx == 0 else "vision_backbone.fused_featurizer"
        for key, value in vision_backbone_state_dict.items():
            hf_state_dict[f"{prefix}.{key}"] = value

    return hf_state_dict


@draccus.wrap()
def convert_prismatic_weights_to_hf(cfg: HFConvertConfig) -> None:
    print(f"[*] Converting Prismatic Model `{cfg.prismatic_model_path_or_id}` to HF Transformers Format")
    torch.set_default_dtype(torch.bfloat16)

    # Get `config.json` and `checkpoint_pt` -- mirrors logic in `prismatic.models.load.py`
    if os.path.isdir(cfg.prismatic_model_path_or_id):
        print(f"[*] Loading from Local Path `{(run_dir := Path(cfg.prismatic_model_path_or_id))}`")
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"

        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        print(f"[*] Downloading Prismatic Checkpoint from HF Hub :: `TRI-ML/{cfg.prismatic_model_path_or_id}`")
        config_json = hf_hub_download("TRI-ML/prismatic-vlms", f"{cfg.prismatic_model_path_or_id}/config.json")
        checkpoint_pt = hf_hub_download(
            "TRI-ML/prismatic-vlms", f"{cfg.prismatic_model_path_or_id}/checkpoints/latest-checkpoint.pt"
        )

    # Load "Native" Config JSON =>> Create LLM Config & Instantiate Tokenizer
    with open(config_json, "r") as f:
        prismatic_config = json.load(f)["model"]

    # Create HF PrismaticConfig (`transformers.PretrainedConfig`)
    hf_config = PrismaticConfig(
        vision_backbone_id=prismatic_config["vision_backbone_id"],
        llm_backbone_id=prismatic_config["llm_backbone_id"],
        arch_specifier=prismatic_config["arch_specifier"],
        image_resize_strategy=prismatic_config["image_resize_strategy"],
        llm_max_length=prismatic_config["llm_max_length"],
        torch_dtype=torch.bfloat16,
    )

    # Instantiate & Add Pad to Tokenizer =>> following `prismatic.models.materialize.get_llm_backbone_and_tokenizer`
    #   TODO (siddk) :: Implement batched generation -- in which case this should set `padding_side = "left"`!
    print("[*] Instantiating and Patching Tokenizer, LLM Config")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_config.hf_llm_id, model_max_length=hf_config.llm_max_length, token=cfg.hf_token, padding_side="right"
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.init_kwargs.pop("add_prefix_space", None)  # Pop to prevent unnecessary warning on reload...
    assert tokenizer.pad_token_id == hf_config.pad_token_id, "Incorrect Pad Token ID!"
    assert len(tokenizer) > hf_config.text_config.vocab_size, "Tokenizer vocabulary must be larger than LLM vocabulary!"

    # Patch LLM Config in `hf_config` with vocab_size (+ `hf_config.pad_to_multiple_of`), pad_token_id + validate
    hf_config.text_config.vocab_size += hf_config.pad_to_multiple_of
    hf_config.text_config.pad_token_id = hf_config.pad_token_id
    hf_config.text_config.torch_dtype = torch.bfloat16
    assert hf_config.text_config.use_cache, "LLM config `use_cache` should be True for inference (set default)!"

    # Create Vision Backbone & Transform =>> following `prismatic.models.materialize.get_vision_backbone_and_transform`
    #   =>> Deviates a bit from existing code; as such, explicitly tested in `tests/test_image_transforms.py`
    print("[*] Loading TIMM Vision Backbone(s) and Image Transform(s) =>> Initializing PrismaticImageProcessor")
    timm_vision_backbones, input_sizes, interpolations, means, stds = [], [], [], [], []
    for idx, timm_model_id in enumerate(hf_config.timm_model_ids):
        timm_vision_backbone = timm.create_model(
            timm_model_id,
            pretrained=True,
            num_classes=0,
            img_size=hf_config.image_sizes[idx],
            act_layer=hf_config.timm_override_act_layers[idx],
        )
        timm_vision_backbones.append(timm_vision_backbone)

        # Get Per-Backbone Image Processing
        data_cfg = timm.data.resolve_model_data_config(timm_vision_backbone)
        input_sizes.append((3, hf_config.image_sizes[idx], hf_config.image_sizes[idx]))
        interpolations.append(data_cfg["interpolation"])
        means.append(data_cfg["mean"])
        stds.append(data_cfg["std"])

        # Patch `LayerScale` because of HF annoying `fix_key` overwrite...
        for module in timm_vision_backbone.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

    # Create PrismaticImageProcessor (`transformers.ImageProcessingMixin`)
    hf_image_processor = PrismaticImageProcessor(
        use_fused_vision_backbone=hf_config.use_fused_vision_backbone,
        image_resize_strategy=hf_config.image_resize_strategy,
        input_sizes=input_sizes,
        interpolations=interpolations,
        means=means,
        stds=stds,
    )

    # Create top-level PrismaticProcessor (`transformers.ProcessorMixin` =>> enables registry w/ AutoProcessor)
    print("[*] Creating PrismaticProcessor Instance from Tokenizer and PrismaticImageProcessor")
    hf_processor = PrismaticProcessor(image_processor=hf_image_processor, tokenizer=tokenizer)

    # Load Prismatic Model State Dictionary (in preparation for conversion)
    print("[*] Loading Prismatic VLM State Dictionary from Checkpoint")
    model_state_dict = torch.load(checkpoint_pt, map_location="cpu")["model"]
    assert ("downsampler" not in model_state_dict) or (len(model_state_dict["downsampler"]) == 0), "Downsampler?"
    assert ("projector" in model_state_dict) and ("llm_backbone" in model_state_dict), "Missing keys!"

    # Convert
    print("[*] Running Conversion")
    converted_state_dict = remap_state_dicts_for_hf(
        model_state_dict["projector"],
        model_state_dict["llm_backbone"],
        vision_backbone_state_dicts=[vb.state_dict() for vb in timm_vision_backbones],
    )

    # Create PrismaticForConditionalGeneration =>> Note that we can't initialize on `meta` device because TIMM
    print("[*] Building (Randomly Initialized) Model =>> PrismaticForConditionalGeneration")
    hf_model = PrismaticForConditionalGeneration(hf_config)
    hf_model.load_state_dict(converted_state_dict, strict=True, assign=True)

    # Cast Model to BF16 before Saving
    hf_model.to(torch.bfloat16)

    # Save Pretrained Versions to Local Path
    print("[*] Saving Model & Processor to Local Path")
    hf_model.save_pretrained(cfg.output_hf_model_local_path, max_shard_size="7GB")
    hf_image_processor.save_pretrained(cfg.output_hf_model_local_path)
    hf_processor.save_pretrained(cfg.output_hf_model_local_path)

    # Register AutoClasses
    PrismaticConfig.register_for_auto_class()
    PrismaticImageProcessor.register_for_auto_class("AutoImageProcessor")
    PrismaticProcessor.register_for_auto_class("AutoProcessor")
    PrismaticForConditionalGeneration.register_for_auto_class("AutoModelForVision2Seq")

    # Push to Hub
    print("[*] Pushing Model & Processor to HF Hub")
    hf_config.push_to_hub(cfg.output_hf_model_hub_path)
    hf_model.push_to_hub(cfg.output_hf_model_hub_path, max_shard_size="7GB")
    hf_image_processor.push_to_hub(cfg.output_hf_model_hub_path)
    hf_processor.push_to_hub(cfg.output_hf_model_hub_path)


if __name__ == "__main__":
    convert_prismatic_weights_to_hf()
