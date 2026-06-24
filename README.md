the part where you # SignTax — Advertising Sign Size Estimator

Faster R-CNN (ResNet101-FPN) detection + Apple Depth Pro monocular depth to
estimate the real-world size of advertising signs from a single photo.

Model weights are pulled at runtime from the Hugging Face model repo
[`prannanan/SignTax`](https://huggingface.co/prannanan/SignTax).

## Notebooks

- **`Object Detection.ipynb`** — This is where we teach the computer to find signs in a photo.
  We start with a model that already knows how to spot lots of everyday objects, then keep
  training it on our own sign pictures until it gets good at drawing a box around each sign.
  At the end it shows a score for how accurate it is.
- **`Depth_SizeEstimation.ipynb`** — This one figures out how big a sign really is when the photo
  is taken straight on. First it finds the sign, then it uses a tool that guesses how far away
  everything in the photo is. Once we know the distance, a bit of simple camera math turns the
  size in the photo into the real width and height in metres.
- **`TiltSign_SizeEstimation.ipynb`** — Same idea, but for signs that are turned at an angle.
  When a sign is tilted it looks smaller or squished in the photo, so this notebook works out
  the angle it's facing and fixes the measurement so we still get the true size.
- **`Depth&Tilt_SignSizeEstimation.ipynb`** — The full version that does everything: finds the
  sign, guesses the distance, fixes the tilt, and adds one extra accuracy step called
  calibration. It then shows each sign's size **before and after** the fix so you can see how
  much it helped.


## Download ml-depth-pro
https://github.com/prannanan/ml-depth-pro


## Dataset
https://drive.google.com/drive/folders/17tN1kYDgEzkMO94LYGTI23DNo3UEeG5s?usp=sharing