import cv2
import numpy as np

def extract_roi(denoised_tensor):
    img = denoised_tensor.squeeze().cpu().numpy() * 255
    img = img.astype('uint8')

    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output
  def noise(denoised):
    img = denoised_tensor
