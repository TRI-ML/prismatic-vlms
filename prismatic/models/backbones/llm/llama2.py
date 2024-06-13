"""
llama2.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
LLAMA2_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "llama2-7b-pure": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-7b-hf"
    },

    "llama2-13b-pure": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-hf"
    },

    # === Meta LLaMa-2 Chat Models ===
    "llama2-7b-chat": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-7b-chat-hf"
    },

    "llama2-13b-chat": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-chat-hf"
    },

    # === Vicuna v1.5 Chat Models ===
    "vicuna-v15-7b": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-7b-v1.5"
    },

    "vicuna-v15-13b": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-13b-v1.5"
    },
}
# fmt: on


class LLaMa2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **LLAMA2_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("llama2-") and self.identifier.endswith("-pure"):
            return PurePromptBuilder

        elif self.identifier.startswith("llama2-") and self.identifier.endswith("-chat"):
            return LLaMa2ChatPromptBuilder

        elif self.identifier.startswith("vicuna"):
            return VicunaV15ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16
