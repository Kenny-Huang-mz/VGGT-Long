from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from loop_utils.config_utils import load_config

from .bridges import discover_bridge_sets
from .chunking import build_bridge_aware_chunks
from .clustering import build_geometry_clusters
from .descriptors import extract_image_descriptors
from .io_utils import ensure_dir, list_images, write_csv, write_json, write_placeholder
from .types import ChunkGraph, ImageNode
from .view_graph import build_view_graph


@dataclass
class OrderFreePipelineArgs:
    image_dir: str
    output_dir: str
    backbone: str
    max_chunk_size: int
    min_chunk_size: int
    knn: int
    bridge_top_m: int
    mutual_knn: bool
    use_geom_verification: bool
    align_mode: str
    config_path: str
    weight_threshold: float | None = None
    shuffle_seed: int | None = None


def _nodes_payload(nodes: List[ImageNode]) -> List[Dict[str, object]]:
    return [node.to_dict(include_descriptor=False) for node in nodes]


def _build_image_items(image_paths: List[str], shuffle_seed: int | None) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    items = [
        {
            "original_id": idx,
            "path": path,
            "original_sorted_index": idx,
        }
        for idx, path in enumerate(image_paths)
    ]
    order_info = {
        "shuffle_seed": shuffle_seed,
        "original_sorted_ids": [item["original_id"] for item in items],
        "processing_order_original_ids": [item["original_id"] for item in items],
        "shuffled": False,
    }
    if shuffle_seed is None:
        return items, order_info

    rng = random.Random(shuffle_seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    order_info.update(
        {
            "shuffled": True,
            "processing_order_original_ids": [item["original_id"] for item in shuffled],
        }
    )
    return shuffled, order_info


def run_order_free_pipeline(args: OrderFreePipelineArgs) -> Dict[str, object]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = ensure_dir(args.output_dir)
    logs_dir = ensure_dir(os.path.join(output_dir, "logs"))
    ensure_dir(os.path.join(output_dir, "reconstruction"))
    ensure_dir(os.path.join(output_dir, "visualizations"))
    write_placeholder(
        os.path.join(output_dir, "reconstruction", "README.txt"),
        "Priority 1 placeholder. Reconstruction outputs will be added in the next milestone.",
    )
    write_placeholder(
        os.path.join(output_dir, "visualizations", "README.txt"),
        "Priority 1 placeholder. Optional visualizations will be added in the next milestone.",
    )

    config = load_config(args.config_path)
    original_image_paths = list_images(args.image_dir)
    if not original_image_paths:
        raise ValueError(f"No images found in {args.image_dir}")
    image_items, order_info = _build_image_items(original_image_paths, args.shuffle_seed)

    descriptor_result = extract_image_descriptors(
        image_items=image_items,
        output_dir=output_dir,
        repo_root=repo_root,
        vggt_long_config=config,
    )
    nodes = descriptor_result.nodes
    descriptors_by_id = {
        node.id: node.descriptor for node in nodes if node.descriptor is not None
    }
    image_paths_by_id = {node.id: node.path for node in nodes}

    edges, view_graph_stats = build_view_graph(nodes, knn=args.knn, mutual_knn=args.mutual_knn)
    clusters, cluster_stats, fallback_events = build_geometry_clusters(
        num_nodes=len(nodes),
        edges=edges,
        min_cluster_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
        bridge_top_m=args.bridge_top_m,
        weight_threshold=args.weight_threshold,
    )
    node_centrality = {
        node.id: float(sum(edge.weight for edge in edges if edge.i == node.id or edge.j == node.id))
        for node in nodes
    }
    bridge_sets, bridge_stats = discover_bridge_sets(
        clusters=clusters,
        edges=edges,
        num_nodes=len(nodes),
        bridge_top_m=args.bridge_top_m,
    )
    chunk_graph, chunk_stats = build_bridge_aware_chunks(
        clusters=clusters,
        bridge_sets=bridge_sets,
        image_paths_by_id=image_paths_by_id,
        descriptors_by_id=descriptors_by_id,
        node_centrality=node_centrality,
        max_chunk_size=args.max_chunk_size,
    )

    run_config = {
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "backbone": args.backbone,
        "max_chunk_size": args.max_chunk_size,
        "min_chunk_size": args.min_chunk_size,
        "knn": args.knn,
        "bridge_top_m": args.bridge_top_m,
        "mutual_knn": args.mutual_knn,
        "use_geom_verification_requested": args.use_geom_verification,
        "use_geom_verification_effective": False,
        "geom_score_status": "reserved_not_enabled_in_v1",
        "align_mode_requested": args.align_mode,
        "align_mode_effective": "not_executed_in_v1",
        "config_path": os.path.abspath(args.config_path),
        "extractor_name": descriptor_result.extractor_name,
        "fallback_used": descriptor_result.fallback_used,
        "fallback_reason": descriptor_result.fallback_reason,
        "shuffle_seed": args.shuffle_seed,
        "input_order": order_info,
    }

    view_graph_payload = {
        "nodes": _nodes_payload(nodes),
        "edges": [edge.to_dict() for edge in edges],
        "config": {
            "knn": args.knn,
            "mutual_knn": args.mutual_knn,
            "extractor_name": descriptor_result.extractor_name,
            "descriptor_dim": descriptor_result.descriptor_dim,
            "fallback_used": descriptor_result.fallback_used,
            "candidate_retrieval_only": True,
            "temporal_sorting_used": False,
            "shuffle_seed": args.shuffle_seed,
            "input_order_shuffled": bool(order_info["shuffled"]),
        },
        "stats": view_graph_stats,
    }
    write_json(os.path.join(output_dir, "view_graph.json"), view_graph_payload)
    write_json(
        os.path.join(output_dir, "bridge_frames.json"),
        {
            "bridge_sets": [bridge_set.to_dict() for bridge_set in bridge_sets],
            "stats": bridge_stats,
        },
    )
    write_json(
        os.path.join(output_dir, "chunks.json"),
        {
            "chunks": [chunk.to_dict() for chunk in chunk_graph.chunks],
            "notes": {
                "shared_bridge_frames_are_used_for_chunk_overlap": True,
                "output_represents_chunk_graph_not_temporal_sequence": True,
            },
        },
    )
    write_json(
        os.path.join(output_dir, "chunk_graph.json"),
        {
            "chunk_graph": chunk_graph.to_dict(),
            "bridge_sets": [bridge_set.to_dict() for bridge_set in bridge_sets],
            "notes": {
                "sim3_status": "reserved_not_computed_in_v1",
                "residual_status": "reserved_not_computed_in_v1",
                "chunk_graph_not_temporal_sequence": True,
            },
        },
    )
    write_json(os.path.join(logs_dir, "run_config.json"), run_config)
    write_json(
        os.path.join(logs_dir, "fallbacks.json"),
        {
            "descriptor_events": descriptor_result.events,
            "graph_events": fallback_events,
        },
    )
    write_json(
        os.path.join(logs_dir, "cluster_stats.json"),
        {
            "cluster_stats": cluster_stats,
            "clusters": [cluster.to_dict() for cluster in clusters],
        },
    )

    csv_rows = []
    for edge in edges:
        csv_rows.append(
            {
                "i": edge.i,
                "j": edge.j,
                "path_i": image_paths_by_id[edge.i],
                "path_j": image_paths_by_id[edge.j],
                "app_score": edge.app_score,
                "geom_score": edge.geom_score,
                "weight": edge.weight,
                "mutual_knn": edge.metadata.get("mutual_knn"),
                "extractor": edge.metadata.get("extractor"),
                "fallback_used": edge.metadata.get("fallback_used"),
            }
        )
    write_csv(
        os.path.join(output_dir, "edge_scores.csv"),
        ["i", "j", "path_i", "path_j", "app_score", "geom_score", "weight", "mutual_knn", "extractor", "fallback_used"],
        csv_rows,
    )

    summary = {
        "method": "order_free_reconstruction_mvp_v1",
        "num_images": len(nodes),
        "descriptor_extractor_used": descriptor_result.extractor_name,
        "descriptor_fallback_used": descriptor_result.fallback_used,
        "candidate_edge_count": view_graph_stats["candidate_edge_count"],
        "kept_edge_count": view_graph_stats["kept_edge_count"],
        "num_clusters": len(clusters),
        "cluster_sizes": cluster_stats["cluster_sizes"],
        "num_bridge_sets": len(bridge_sets),
        "num_chunks": len(chunk_graph.chunks),
        "chunk_sizes": chunk_stats["chunk_sizes"],
        "chunk_edge_count": len(chunk_graph.edges),
        "shared_frame_counts": chunk_stats["shared_frame_counts"],
        "use_geom_verification_effective": False,
        "align_mode_effective": "not_executed_in_v1",
        "notes": [
            "image similarity is only used for candidate retrieval",
            "geometry verification is reserved for a later milestone",
            "bridge frames are shared by multiple chunks",
            "output is a chunk graph, not a recovered temporal sequence",
        ],
    }
    write_json(os.path.join(logs_dir, "summary.json"), summary)
    return summary
