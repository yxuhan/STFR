import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
from runtime.soft_foreground_segmenter import SoftForegroundSegmenter


def main():
    """Main function to run the demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Demo script for depth estimation, foreground segmentation, and surface normal estimation"
    )
    parser.add_argument("--input_root", required=True, help="Path to input image")
    parser.add_argument(
        "--foreground-model", help="Path to foreground segmentation ONNX model", 
        default="model/foreground-segmentation-model-vitl16_384.onnx"
    )

    parser.add_argument("--output_root", help="Save result to a path (optional)")

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    foreground_segmenter = SoftForegroundSegmenter(onnx_model=args.foreground_model)

    for pth in tqdm(sorted(os.listdir(args.input_root))):
        image = cv2.imread(os.path.join(args.input_root, pth))
        
        fg = foreground_segmenter.estimate_foreground_segmentation(
            image
        )
        cv2.imwrite(
            os.path.join(args.output_root, pth), 
            (fg[..., None] * 255.).astype(np.uint8)
        )


if __name__ == "__main__":
    main()
