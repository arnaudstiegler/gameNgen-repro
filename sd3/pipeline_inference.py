from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
import torch
from time import time
import diffusers
import numpy as np
import random

diffusers.logging.set_verbosity_info()
torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)


generator = torch.Generator()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# scheduler =  DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
scheduler =  DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
).to(device)

prompt = "Spiderman riding a horse"

start = time()
image = pipe(prompt).images[0]
print(f'Took {time() - start} with device {device}')
image.save("astronaut_rides_horse.png")

