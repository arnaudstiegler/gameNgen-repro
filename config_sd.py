# Number of past frames/actions we provide
# Beware that there's a hard limit coming from the dataset
BUFFER_SIZE = 9
# Given by the paper
ZERO_OUT_ACTION_CONDITIONING_PROB = 0.1

HEIGHT = 240
WIDTH = 320

# CFG ratio
CFG_GUIDANCE_SCALE = 1.5

# Conditional Image noise parameters
# Those values are the same as in the paper
MAX_NOISE_LEVEL = 0.7
NUM_BUCKETS = 10

# Repo name for dumping model artifacts
REPO_NAME = "arnaudstiegler/sd-model-gameNgen"

# When not using frame conditioning, we use this prompt
VALIDATION_PROMPT = "video game doom image, high quality, 4k, high resolution"

# Datasets
TRAINING_DATASET_DICT = {
    "small": "P-H-B-D-a16z/ViZDoom-Deathmatch-PPO",
    "large": "P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-Lrg",
}