import cv2
import torch

def load_image(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = torch.tensor(img).float().unsqueeze(0)  # (1,H,W)
    return img.unsqueeze(0)  # (B,1,H,W)
