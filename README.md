# SignTax — Advertising Sign Size Estimator

Faster R-CNN (ResNet101-FPN) detection + Apple Depth Pro monocular depth to
estimate the real-world size of advertising signs from a single photo.

Model weights are pulled at runtime from the Hugging Face model repo
[`prannanan/SignTax`](https://huggingface.co/prannanan/SignTax).

## Notebooks

- **`Object Detection.ipynb`** — Trains the sign detector via transfer learning: loads a
  COCO-pretrained Faster R-CNN (ResNet101-FPN) backbone, replaces the classification head
  for the custom sign classes, fine-tunes on the dataset, and evaluates mAP / IoU.
- **`Depth_SizeEstimation.ipynb`** — End-to-end size estimation for frontal signs. Runs the
  trained detector to get bounding boxes, runs Apple Depth Pro for a metric depth map and
  focal length, then applies the pinhole camera model (`W = w·Z / f`) to estimate each
  sign's real-world width and height in metres.
- **`TiltSign_SizeEstimation.ipynb`** — Size estimation for *tilted* signs. Adds RANSAC
  plane-fitting on the depth mask to recover the sign's surface normal and corrects for
  foreshortening (`1/cos(tilt angle)`), so angled signs are measured accurately.
- **`Depth&Tilt_SignSizeEstimation.ipynb`** — The full end-to-end pipeline that combines the
  tilt correction with calibration. For each sign it runs detector → Depth Pro → RANSAC
  plane-fitting → foreshortening correction to get a *tilt-corrected* size, then fits a single
  calibration factor `K = median(true / predicted)` on the train+valid splits and multiplies it
  in. The notebook reports every sign's size **before vs after calibration** on the held-out
  test split — as a per-sign table, regression metrics (MAE / RMSE / MAPE / R² / % within ±10%),
  and side-by-side images — so the effect of calibration is visible at a glance.


## Download ml-depth-pro
https://github.com/prannanan/ml-depth-pro


## Dataset
https://drive.google.com/drive/folders/17tN1kYDgEzkMO94LYGTI23DNo3UEeG5s?usp=sharing