import requests
import torch
import os

from PIL import Image
from pathlib import Path
from prismatic import load
from prismatic.models.backbones.vision.vggt import VGGTBackbone
from vggt.models.vggt import VGGT

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = ""
device = "cuda"

# Path to original model
original_model_id = "dinov2-224px+7b"  # Replace with your original model ID or path
# Path to your fine-tuned checkpoint that has only vggt_projector
finetuned_checkpoint_path = "/home/scur0690/prismatic-vlms/runs/OpenSpaces_MC+dinov2-224px+7b+stage-finetune+x42"

# 1. Load the original model
vlm = load(original_model_id)#, hf_token=hf_token)

# 2. Add VGGT backbone
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device, dtype=torch.bfloat16)
vggt_backbone = VGGTBackbone(vggt_model)
vlm.vggt_backbone = vggt_backbone
vlm.vggt_backbone.to(device, dtype=torch.bfloat16)


# 4. Now load just the vggt_projector frsom your fine-tuned checkpoint
checkpoint_file = Path(finetuned_checkpoint_path) / "checkpoints" / "latest-checkpoint.pt"

if checkpoint_file.exists():
    print(f"Loading vggt_projector from {checkpoint_file}")
    model_state_dict = torch.load(checkpoint_file, map_location=device)["model"]
    if "vggt_projector" in model_state_dict:
        vlm.vggt_projector.load_state_dict(model_state_dict["vggt_projector"])
        print("Successfully loaded vggt_projector weights from checkpoint")
    else:
        print(f"Warning: vggt_projector not found in checkpoint. Available keys: {list(model_state_dict.keys())}")
else:
    print(f"Warning: Checkpoint file {checkpoint_file} does not exist")

# Set vggt_projector for inference
vlm.vggt_projector_requires_grad = False

print(f"Model state dict keys: {list(vlm.state_dict().keys())}")
# Move model to device
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# # save image
# image.save("image.png")
image_path = r"/home/scur0690/prismatic-vlms/data/OpenSpaces_MC/test/images/0.jpg"
image = Image.open(image_path).convert("RGB")

user_prompt = "What is the horizontal distance between the stove and the fridge?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()
print(prompt_text)

# Generate!
with torch.inference_mode():
    generated_text = vlm.generate(
        image=image,
        prompt_text=prompt_text,
        image_paths=image_path,  # Add image path for VGGT
        do_sample=True,
        temperature=0.4,
        max_new_tokens=512,
        min_length=8,
    )
print(generated_text)
