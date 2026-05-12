import argparse
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from order_free.pipeline import OrderFreePipelineArgs, run_order_free_pipeline


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the order-free chunk-graph reconstruction MVP.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing unordered input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save graph/chunk artifacts.")
    parser.add_argument("--backbone", type=str, default="pi3", choices=["pi3", "vggt"], help="Reserved for future local reconstruction stages.")
    parser.add_argument("--max_chunk_size", type=int, default=80, help="Maximum images per chunk, including shared bridge frames.")
    parser.add_argument("--min_chunk_size", type=int, default=20, help="Preferred minimum cluster core size before weak-cluster fallback.")
    parser.add_argument("--knn", type=int, default=10, help="Top-k nearest neighbors for candidate view-graph construction.")
    parser.add_argument("--bridge_top_m", type=int, default=12, help="Maximum number of bridge frames to share between adjacent clusters.")
    parser.add_argument("--mutual_knn", type=str2bool, default=True, help="Whether to keep only mutual-kNN candidate edges.")
    parser.add_argument("--use_geom_verification", action="store_true", help="Reserved flag. Logged but not executed in MVP V1.")
    parser.add_argument("--align_mode", type=str, default="graph", help="Reserved flag. Logged but not executed in MVP V1.")
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "base_config.yaml"), help="VGGT-Long config path used to locate optional descriptor weights.")
    parser.add_argument("--weight_threshold", type=float, default=None, help="Optional override for cluster graph edge threshold.")
    args = parser.parse_args()

    pipeline_args = OrderFreePipelineArgs(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        knn=args.knn,
        bridge_top_m=args.bridge_top_m,
        mutual_knn=args.mutual_knn,
        use_geom_verification=args.use_geom_verification,
        align_mode=args.align_mode,
        config_path=args.config,
        weight_threshold=args.weight_threshold,
    )

    summary = run_order_free_pipeline(pipeline_args)
    print("Order-free reconstruction MVP V1 finished.")
    print(f"Images: {summary['num_images']}, clusters: {summary['num_clusters']}, chunks: {summary['num_chunks']}")
    print(f"Descriptor extractor: {summary['descriptor_extractor_used']} (fallback={summary['descriptor_fallback_used']})")


if __name__ == "__main__":
    main()
