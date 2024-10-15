import torch


def no_img_conditioning_augmentation(
    images: torch.Tensor,
    prob: float = 0.0
) -> torch.Tensor:
    """
    Zeroes out the conditioning frames with probability `prob`.
    This is necessary to train the model on no frame conditioning,
    allowing for unconditional generation with CFG.
    """
    turn_off_conditioning_prob = torch.rand(images.shape[0],
                                            1,
                                            1,
                                            1,
                                            device=images.device)
    no_img_conditioning_prob = turn_off_conditioning_prob < prob
    no_img_conditioning_prob = no_img_conditioning_prob.unsqueeze(1).expand(
        -1, images.shape[1] - 1, -1, -1, -1)
    images[:, :-1] = torch.where(no_img_conditioning_prob,
                                 torch.zeros_like(images[:, :-1]),
                                 images[:, :-1])
    return images
