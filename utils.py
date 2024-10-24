import torch
from config_sd import MAX_NOISE_LEVEL, NUM_BUCKETS


def discretize_noise_level(noise_level: float) -> int:
    size_bucket = torch.tensor(MAX_NOISE_LEVEL / NUM_BUCKETS)
    return torch.min(torch.round(noise_level / size_bucket), torch.tensor(NUM_BUCKETS - 1)).to(torch.int32)


def get_conditioning_noise(conditioning_frames: torch.Tensor) -> torch.Tensor:
    """
    Get the noise level for the conditioning frames


    Args:
        conditioning_frames: the conditioning frames, shape (bs, buffer_size, 3, 64, 64)

    Returns:
        discretized_noise_level: the discretized noise level
    """
    noise_level = torch.rand(conditioning_frames.shape[0], device=conditioning_frames.device) * MAX_NOISE_LEVEL
    discretized_noise_level = discretize_noise_level(noise_level)
    return noise_level, discretized_noise_level

def add_conditioning_noise(conditioning_frames: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
    """
    Add Gaussian noise to the conditioning frames
    """
    conditional_noise = torch.randn_like(conditioning_frames) * noise_level.view(-1, 1, 1, 1, 1)
    noisy_conditioning_frames = conditioning_frames + conditional_noise
    return noisy_conditioning_frames


if __name__ == "__main__":
    conditioning_noise_embeddings = torch.randn(NUM_BUCKETS, 16)
    conditioning_frames = torch.randn(2, 3, 4, 64, 64)
    noise_level, discretized_noise_level = get_conditioning_noise(conditioning_frames)
    
    # Add Gaussian noise to the conditioning frames
    noisy_conditioning_frames = add_conditioning_noise(conditioning_frames, noise_level)
    print(noise_level)
    print(discretized_noise_level)
    print(conditioning_noise_embeddings[discretized_noise_level])