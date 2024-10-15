# Number of past frames/actions we provide
BUFFER_SIZE = 2
# Given by the paper
ZERO_OUT_ACTION_CONDITIONING_PROB = 0.1

# TODO: long-term, we should support arbitrary aspect ratios
# HEIGHT = 240
# WIDTH = 320

# For now, we only support square images (default size for sd models)
WIDTH = HEIGHT = 512

# Repo name for dumping model artifacts
REPO_NAME = "arnaudstiegler/sd-model-gameNgen"

VALIDATION_PROMPT = "video game doom image, high quality, 4k, high resolution"