import argparse
import glob
import os
import shutil
from datetime import datetime

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import warmup_numba
from vggt_long import VGGT_Long, copy_file


def build_save_dir(image_dir, out_path, save_dir=None):
    if save_dir is not None:
        return save_dir

    base_name = os.path.splitext(os.path.basename(out_path))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return os.path.join(os.path.dirname(os.path.abspath(out_path)), f"{base_name}_artifacts_{timestamp}")


def list_images(image_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    image_names = []
    for ext in exts:
        image_names.extend(glob.glob(os.path.join(image_dir, ext)))
        image_names.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(image_names)


def select_image_range(image_names, begin=None, end=None):
    if begin is None and end is None:
        return image_names

    if begin is None or end is None:
        raise ValueError("--begin and --end must be specified together")
    if begin < 1:
        raise ValueError("--begin must be >= 1")
    if end < begin:
        raise ValueError("--end must be >= --begin")
    if end > len(image_names):
        raise ValueError(f"--end out of range: {end}, total images: {len(image_names)}")

    selected = image_names[begin - 1 : end]
    print(f"Selecting frames {begin} to {end}, total {end - begin + 1}")
    return selected


def materialize_selected_images(image_paths, save_dir):
    selected_dir = os.path.join(save_dir, "_selected_inputs")
    os.makedirs(selected_dir, exist_ok=True)

    for idx, src_path in enumerate(image_paths):
        ext = os.path.splitext(src_path)[1].lower()
        dst_name = f"{idx:06d}{ext}"
        dst_path = os.path.join(selected_dir, dst_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        try:
            os.symlink(os.path.abspath(src_path), dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)

    return selected_dir


def main():
    parser = argparse.ArgumentParser(description="Run VGGT-Long and save the final aligned reconstruction as .npz.")
    parser.add_argument("--image_dir", type=str, required=True, help="Input image directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/base_config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument("--out_path", type=str, required=True, help="Path to save the .npz result.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory for intermediate VGGT-Long artifacts.",
    )
    parser.add_argument("--max_images", type=int, default=None, help="Use only the first N input images.")
    parser.add_argument("--begin", type=int, default=None, help="Select start frame index, 1-based.")
    parser.add_argument("--end", type=int, default=None, help="Select end frame index, 1-based.")
    args = parser.parse_args()

    config = load_config(args.config)
    save_dir = build_save_dir(args.image_dir, args.out_path, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    copy_file(args.config, save_dir)

    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    image_paths = select_image_range(image_paths, begin=args.begin, end=args.end)
    run_image_dir = materialize_selected_images(image_paths, save_dir)
    print(f"Prepared {len(image_paths)} input images under {run_image_dir}")

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    runner = VGGT_Long(run_image_dir, save_dir, config)
    try:
        runner.run()
        runner.export_reconstruction_npz(args.out_path)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
