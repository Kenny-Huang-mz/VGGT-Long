import argparse
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from order_free.pipeline import OrderFreePipelineArgs, run_order_free_pipeline
from order_free.reconstruction import ReconstructionArgs, run_priority2_reconstruction, validate_backbone_config


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _priority1_outputs_exist(output_dir: str) -> bool:
    required = [
        os.path.join(output_dir, "chunks.json"),
        os.path.join(output_dir, "chunk_graph.json"),
        os.path.join(output_dir, "bridge_frames.json"),
    ]
    return all(os.path.exists(path) for path in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full order-free reconstruction pipeline (Priority 1 + Priority 2).")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing unordered input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save graph, chunk, and reconstruction artifacts.")
    parser.add_argument("--backbone", type=str, default="pi3", choices=["pi3", "vggt"], help="Priority 2 local reconstruction backbone.")
    parser.add_argument("--max_chunk_size", type=int, default=80, help="Maximum images per chunk, including shared bridge frames.")
    parser.add_argument("--min_chunk_size", type=int, default=20, help="Preferred minimum cluster core size before weak-cluster fallback.")
    parser.add_argument("--knn", type=int, default=10, help="Top-k nearest neighbors for candidate view-graph construction.")
    parser.add_argument("--bridge_top_m", type=int, default=12, help="Maximum number of bridge frames to share between adjacent clusters.")
    parser.add_argument("--mutual_knn", type=str2bool, default=True, help="Whether to keep only mutual-kNN candidate edges.")
    parser.add_argument("--use_geom_verification", action="store_true", help="Reserved flag. Logged but not executed in this milestone.")
    parser.add_argument("--align_mode", type=str, default="graph_mst", help="Priority 2 alignment mode. Only graph_mst is supported.")
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "base_config.yaml"), help="VGGT-Long config path.")
    parser.add_argument("--weight_threshold", type=float, default=None, help="Optional override for cluster graph edge threshold.")
    parser.add_argument("--shuffle_seed", type=int, default=None, help="Optional seed to shuffle input processing order while preserving original sorted image ids in Priority 1 outputs.")
    parser.add_argument("--skip_priority1_if_exists", action="store_true", help="Reuse existing chunks.json/chunk_graph.json if they already exist in output_dir.")
    parser.add_argument("--chunk_cache_dir", type=str, default=None, help="Optional directory for chunk-level Pi3 prediction caches.")
    parser.add_argument("--pointcloud_sample_ratio", type=float, default=1.0, help="Random sampling ratio applied after confidence filtering when exporting the merged point cloud.")
    parser.add_argument("--pointcloud_conf_quantile", type=float, default=0.7, help="Per-chunk confidence quantile threshold used to filter points before merging.")
    args = parser.parse_args()
    from loop_utils.config_utils import load_config

    config = load_config(args.config)
    weights_key_used = validate_backbone_config(args.backbone, config)
    print(f"Selected backbone: {args.backbone}")
    print(f"Config path: {os.path.abspath(args.config)}")
    print(f"Weights key used: {weights_key_used}")

    if args.skip_priority1_if_exists and _priority1_outputs_exist(args.output_dir):
        priority1_summary = None
        print("Reusing existing Priority 1 artifacts from output_dir.")
    else:
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
            shuffle_seed=args.shuffle_seed,
        )
        priority1_summary = run_order_free_pipeline(pipeline_args)
        print("Priority 1 finished.")

    reconstruction_args = ReconstructionArgs(
        output_dir=args.output_dir,
        config_path=args.config,
        backbone=args.backbone,
        align_mode=args.align_mode,
        chunk_cache_dir=args.chunk_cache_dir,
        pointcloud_sample_ratio=args.pointcloud_sample_ratio,
        pointcloud_conf_quantile=args.pointcloud_conf_quantile,
    )
    reconstruction_summary = run_priority2_reconstruction(reconstruction_args)

    print("Order-free reconstruction MVP V2 finished.")
    if priority1_summary is not None:
        print(
            f"Priority 1: images={priority1_summary['num_images']}, "
            f"clusters={priority1_summary['num_clusters']}, chunks={priority1_summary['num_chunks']}"
        )
    print(
        f"Priority 2: chunk_predictions={reconstruction_summary['num_successful_chunk_predictions']}, "
        f"aligned_edges={reconstruction_summary['aligned_edge_count']}, "
        f"components={reconstruction_summary['component_count']}, "
        f"merged_points={reconstruction_summary['merged_point_count']}"
    )


if __name__ == "__main__":
    main()
