# 🌿 Palm Leaf Manuscript Denoising & ROI Extraction

A deep learning pipeline to denoise degraded palm leaf manuscript images and extract the text region using U-Net and OpenCV.

Built as part of **VTLab IIT Tirupati Summer Internship 2026**.

---

## 📌 Problem Statement

Palm leaf manuscripts are ancient documents that have become very damaged over time. They have:
- Yellowing and brown discoloration
- Stains and water damage
- Noise and fading ink
- Insect holes and physical tears

**Goal:** Given a noisy/degraded palm leaf image → produce a clean denoised image → highlight the text region with bounding boxes.

---

## 🗂️ Project Structure

```
palm-leaf-denoising/
│
├── data/
│   └── sample_input.jpg       # Input palm leaf image
│
├── outputs/
│   ├── noisy.jpg              # Noisy input image
│   ├── denoised.jpg           # U-Net denoised output
│   ├── roi.jpg                # ROI with bounding boxes
│   └── full_pipeline.png      # Side by side result
│
├── unet.py                    # U-Net model definition
├── dataset.py                 # Image loading and preprocessing
├── noise.py                   # Synthetic Gaussian noise
├── roi.py                     # ROI extraction using OpenCV
├── train.py                   # Training loop
├── infer.py                   # Inference and saving outputs
└── README.md
```

---

## ⚙️ Approach

I used a **two-stage pipeline**:

### Stage 1 — U-Net Denoising
- U-Net is an encoder-decoder CNN with skip connections
- Skip connections preserve fine script details during reconstruction
- Input: noisy 256x256 grayscale image
- Output: clean denoised image
- Since no labeled dataset was available, I used **synthetic Gaussian noise** to create (noisy, clean) training pairs

### Stage 2 — ROI Extraction (OpenCV)
- Apply binary thresholding (value = 150) to separate text from background
- Use `cv2.findContours` to detect text regions
- Draw green bounding boxes around each detected region

---

## 🧠 Model Architecture

```
Input (1x256x256)
      │
  [Encoder]
  Conv Block 1 → 64 ch  → MaxPool
  Conv Block 2 → 128 ch → MaxPool
  Conv Block 3 → 256 ch → MaxPool
      │
  [Bottleneck]
  Conv Block   → 512 ch
      │
  [Decoder]
  Upsample + Skip → 256 ch
  Upsample + Skip → 128 ch
  Upsample + Skip → 64 ch
      │
  Final Conv (1x1) + Sigmoid
      │
Output (1x256x256)
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/palm-leaf-denoising.git
cd palm-leaf-denoising
```

### 2. Install dependencies
```bash
pip install torch torchvision opencv-python scikit-image matplotlib
```

### 3. Add your input image
Place any palm leaf image inside the `data/` folder and name it `sample_input.jpg`

### 4. Train the model
```bash
python train.py
```
Output:
```
Epoch 1, Loss: 0.4231
Epoch 2, Loss: 0.3812
...
Epoch 30, Loss: 0.0423
```

### 5. Run inference
```bash
python infer.py
```
Outputs are saved in the `outputs/` folder.

---

## 📊 Results

The model was trained for **30 epochs** on Google Colab (T4 GPU).  
Loss reduced from **0.42 → 0.04**, showing the model learned to remove noise effectively.

### Pipeline Output

| Noisy Input | U-Net Denoised | ROI Extracted |
|:-----------:|:--------------:|:-------------:|
| ![noisy](outputs/noisy.jpg) | ![denoised](outputs/denoised.jpg) | ![roi](outputs/roi.jpg) |

### Full Pipeline

![Full Pipeline](outputs/full_pipeline.png)

> Left: Noisy Input &nbsp;|&nbsp; Center: U-Net Denoised &nbsp;|&nbsp; Right: ROI with Bounding Boxes

---

## 🛠️ Training Details

| Parameter | Value |
|-----------|-------|
| Model | U-Net |
| Input size | 256 x 256 (grayscale) |
| Loss function | L1 Loss |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Epochs | 30 |
| Noise type | Gaussian (factor = 0.2) |
| Platform | Google Colab T4 GPU |

---

## ⚠️ Limitations

- Trained on only one image with synthetic noise
- Simple Gaussian noise does not fully replicate real manuscript degradation
- Basic thresholding may not work well on very heavily damaged manuscripts
- Not tested on specific Indic scripts like Telugu, Sharada or Devanagari

---

## 🔭 Future Work

- Collect real labeled palm leaf dataset for better training
- Add more realistic degradation (yellowing, stains, uneven lighting)
- Try GAN-based model (pix2pix) for sharper results
- Extend pipeline with line segmentation and automatic transcription

---

## 📚 References

- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Prasad et al. (2023) — Damage Detection in Palm Leaf Manuscripts, npj Heritage Science
- OpenCV Docs — https://docs.opencv.org
- PyTorch Docs — https://pytorch.org/docs

---

## 👤 Author

**Rahul Rathod**  
IISER Bhopal  
rathod24@iiserb.ac.in  
VTLab IIT Tirupati — Summer Internship 2026
