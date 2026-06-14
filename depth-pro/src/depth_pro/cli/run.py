#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def run(args):
    """Run Depth Pro on a sample image."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model.
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    image_paths = [args.image_path]
    if args.image_path.is_dir():
        image_paths = args.image_path.glob("**/*")
        relative_path = args.image_path
    else:
        relative_path = args.image_path.parent

    if not args.skip_display:
        plt.ion()
        fig = plt.figure()
        ax_rgb = fig.add_subplot(121)
        ax_disp = fig.add_subplot(122)

    for image_path in tqdm(image_paths):
        # Load image and focal length from exif info (if found.).
        try:
            LOGGER.info(f"Loading image {image_path} ...")
            image, _, f_px = load_rgb(image_path)
            print("FOCAL LENGTH:", f_px)
            print("INPUT FOCAL:", f_px)
            
        except Exception as e:
            LOGGER.error(str(e))
            continue
        # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
        # otherwise the model estimates `f_px` to compute the depth metricness.
        prediction = model.infer(transform(image), f_px=f_px)
        print("MODEL FOCAL:", prediction["focallength_px"])

        # Extract the depth and focal length.
        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        print("MIN DEPTH:", depth.min())
        print("MAX DEPTH:", depth.max())

        if f_px is not None:
            LOGGER.debug(f"Focal length (from exif): {f_px:0.2f}")
        elif prediction["focallength_px"] is not None:
            focallength_px = prediction["focallength_px"].detach().cpu().item()
            LOGGER.info(f"Estimated focal length: {focallength_px}")

        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )

        # Save Depth as npz file.
        if args.output_path is not None:
            output_file = (
                args.output_path
                / image_path.relative_to(relative_path).parent
                / image_path.stem
            )
            LOGGER.info(f"Saving depth map to: {str(output_file)}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, depth=depth)

            # Save as color-mapped "turbo" jpg image.
            cmap = plt.get_cmap("turbo")
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                np.uint8
            )
            color_map_output_file = str(output_file) + ".jpg"
            LOGGER.info(f"Saving color-mapped depth to: : {color_map_output_file}")
            PIL.Image.fromarray(color_depth).save(
                color_map_output_file, format="JPEG", quality=90
            )

        # Display the image and estimated depth map.
        if not args.skip_display:
            ax_rgb.imshow(image)
            ax_disp.imshow(inverse_depth_normalized, cmap="turbo")

            # Show the metric distance (in meters) of the pixel under the cursor.
            h, w = depth.shape

            def on_move(event, depth=depth, h=h, w=w):
                if event.inaxes in (ax_rgb, ax_disp) and event.xdata is not None:
                    col = int(round(event.xdata))
                    row = int(round(event.ydata))
                    if 0 <= row < h and 0 <= col < w:
                        dist = depth[row, col]
                        fig.suptitle(
                            f"Distance at ({col}, {row}): {dist:.3f} m"
                        )
                    else:
                        fig.suptitle("")
                else:
                    fig.suptitle("")
                fig.canvas.draw_idle()

            # Disconnect any handler from a previous image before connecting a new one.
            cid = getattr(fig, "_hover_cid", None)
            if cid is not None:
                fig.canvas.mpl_disconnect(cid)
            fig._hover_cid = fig.canvas.mpl_connect("motion_notify_event", on_move)

            fig.canvas.draw()
            fig.canvas.flush_events()

    LOGGER.info("Done predicting depth!")
    if not args.skip_display:
        plt.show(block=True)


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/example.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Skip matplotlib display.",
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show verbose output."
    )
    
    run(parser.parse_args())


if __name__ == "__main__":
    main()
