import torch

def add_noise(img, noise_factor=0.2):
    noise = torch.randn_like(img) * noise_factor
    noisy = img + noise
    return torch.clamp(noisy, 0., 1.)
