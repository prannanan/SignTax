from PIL import Image
import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb('C:/Users/punda/Desktop/SynTax/before AI Builders/new dataset/no size/S__87769108_0.jpg')
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

import numpy as np
print("Depth shape:", depth.shape)
print("Depth min/max (m):", depth.min().item(), "/", depth.max().item())
print("Focal length (px):", focallength_px.item() if focallength_px is not None else f_px)