import argparse
import os
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
    args = parser.parse_args()

    config = load_config(args.config)
    save_dir = build_save_dir(args.image_dir, args.out_path, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    copy_file(args.config, save_dir)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    runner = VGGT_Long(args.image_dir, save_dir, config)
    try:
        runner.run()
        runner.export_reconstruction_npz(args.out_path)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
