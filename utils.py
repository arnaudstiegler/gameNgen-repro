import torch

# This spec is from the paper
MAX_NOISE_LEVEL = 0.7
NUM_BUCKETS = 10

def discretize_noise_level(noise_level: float) -> int:
    size_bucket = torch.tensor(MAX_NOISE_LEVEL / NUM_BUCKETS)
    return torch.min(torch.round(noise_level / size_bucket), torch.tensor(NUM_BUCKETS - 1)).to(torch.int32)


def get_conditioning_noise(conditioning_frames: torch.Tensor) -> torch.Tensor:
    noise_level = torch.rand(conditioning_frames.shape[0], device=conditioning_frames.device) * MAX_NOISE_LEVEL
    discretized_noise_level = discretize_noise_level(noise_level)
    return noise_level,discretized_noise_level


if __name__ == "__main__":
    conditioning_noise_embeddings = torch.randn(NUM_BUCKETS, 16)
    conditioning_frames = torch.randn(2, 3, 4, 64, 64)
    noise_level, discretized_noise_level = get_conditioning_noise(conditioning_frames)
    print(noise_level)
    print(discretized_noise_level)
    print(conditioning_noise_embeddings[discretized_noise_level])