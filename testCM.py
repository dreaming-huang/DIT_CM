import torch
import os
from torchvision.utils import save_image

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import ConsistencyModelPipeline

device = "cuda"
# Load the cd_imagenet64_l2 checkpoint.
model_id_or_path = "openai/diffusers-cd_cat256_lpips"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Onestep Sampling
image = pipe(num_inference_steps=1).images[0]
image.save("consistency_model_onestep_sample.png")

# Onestep sampling, class-conditional image generation
# ImageNet-64 class label 145 corresponds to king penguins

class_id = 145
class_id = torch.tensor(class_id, dtype=torch.long)

image = pipe(num_inference_steps=1, class_labels=class_id).images[0]
image.save("consistency_model_onestep_sample_penguin.png")

# Multistep sampling, class-conditional image generation
# Timesteps can be explicitly specified; the particular timesteps below are from the original Github repo.
# https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
image = pipe(timesteps=[22, 0], class_labels=class_id).images[0]
image.save("consistency_model_multistep_sample_penguin.png")