from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .alignment import (
    apply_sim3_to_camera_poses,
    apply_sim3_to_points,
    build_chunk_alignments,
    build_global_chunk_poses,
)
from .io_utils import stable_hash, write_json
from .reconstruction_io import (
    chunk_prediction_cache_info,
    load_chunk_graph_from_output,
    load_chunk_prediction,
    merge_pointclouds,
    reconstruction_dirs,
    save_chunk_alignments,
    save_chunk_prediction,
    save_components,
    save_global_poses,
    save_merged_pointcloud,
)
from .types import Chunk, ChunkGraph, ChunkPrediction


@dataclass
class ReconstructionArgs:
    output_dir: str
    config_path: str
    backbone: str
    align_mode: str
    chunk_cache_dir: str | None = None
    pointcloud_sample_ratio: float = 1.0
    pointcloud_conf_quantile: float = 0.7


def validate_backbone_config(backbone: str, config: Dict[str, object]) -> str:
    weights = config.get("Weights", {})
    if backbone == "pi3":
        weight_key = "Pi3"
    elif backbone == "vggt":
        weight_key = "VGGT"
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    weight_path = weights.get(weight_key)
    if not weight_path:
        raise ValueError(f"--backbone {backbone} requires Weights.{weight_key} in the config")
    return weight_key


def build_model_adapter(backbone: str, config: Dict[str, object]):
    if backbone == "pi3":
        from base_models.base_model import Pi3Adapter

        return Pi3Adapter(config)
    if backbone == "vggt":
        from base_models.base_model import VGGTAdapter

        return VGGTAdapter(config)
    raise ValueError(f"Unsupported backbone: {backbone}")


def _config_hash(config_path: str, config: Dict[str, object]) -> str:
    payload = {"config_path": os.path.abspath(config_path), "config": config}
    return stable_hash(payload)


def _chunk_image_paths(chunk: Chunk) -> List[str]:
    image_paths = chunk.metadata.get("image_paths", [])
    if len(image_paths) != len(chunk.image_ids):
        raise ValueError(f"Chunk {chunk.id} is missing metadata.image_paths aligned with image_ids")
    return [str(path) for path in image_paths]


def _to_chunk_prediction(chunk: Chunk, prediction_dict: Dict[str, object], cache_key: str) -> ChunkPrediction:
    prediction = {}
    for key, value in prediction_dict.items():
        if isinstance(value, np.ndarray):
            array = value
        else:
            array = np.asarray(value)
        prediction[key] = array
    if "depth" in prediction:
        prediction["depth"] = np.squeeze(prediction["depth"])

    return ChunkPrediction(
        chunk_id=chunk.id,
        image_ids=list(chunk.image_ids),
        image_paths=_chunk_image_paths(chunk),
        extrinsic=np.asarray(prediction["extrinsic"], dtype=np.float32),
        intrinsic=np.asarray(prediction["intrinsic"], dtype=np.float32),
        world_points=np.asarray(prediction["world_points"], dtype=np.float32),
        world_points_conf=np.asarray(prediction["world_points_conf"], dtype=np.float32),
        images=np.asarray(prediction["images"], dtype=np.float32),
        mask=np.asarray(prediction["mask"], dtype=bool) if prediction.get("mask") is not None else None,
        camera_poses=np.asarray(prediction.get("camera_poses", prediction["extrinsic"]), dtype=np.float32),
        local_points=np.asarray(prediction["local_points"], dtype=np.float32) if prediction.get("local_points") is not None else None,
        conf_prob=np.asarray(prediction["conf_prob"], dtype=np.float32) if prediction.get("conf_prob") is not None else None,
        metadata={
            "cache_key": cache_key,
            "chunk_id": chunk.id,
            "source_cluster_ids": list(chunk.source_cluster_ids),
            "backbone": str(prediction_dict.get("_backbone", "")),
        },
    )


def _global_image_pose_records(
    chunk_graph: ChunkGraph,
    transformed_predictions: Dict[int, ChunkPrediction],
) -> List[Dict[str, object]]:
    best_records: Dict[int, Dict[str, object]] = {}
    chunk_map = {chunk.id: chunk for chunk in chunk_graph.chunks}

    for chunk_id, prediction in transformed_predictions.items():
        chunk = chunk_map[chunk_id]
        pose_source = prediction.camera_poses if prediction.camera_poses is not None else prediction.extrinsic
        conf = prediction.conf_prob if prediction.conf_prob is not None else prediction.world_points_conf
        idx_map = {image_id: idx for idx, image_id in enumerate(prediction.image_ids)}
        bridge_set = set(chunk.bridge_image_ids)
        for image_id, local_idx in idx_map.items():
            frame_conf = float(np.mean(conf[local_idx]))
            is_core = image_id in set(chunk.core_image_ids)
            candidate_score = (1 if is_core else 0, frame_conf, -chunk_id)
            current = best_records.get(image_id)
            if current is None or candidate_score > current["_score"]:
                best_records[image_id] = {
                    "image_id": image_id,
                    "chunk_id": chunk_id,
                    "pose": np.asarray(pose_source[local_idx], dtype=np.float32).tolist(),
                    "is_core_frame": bool(is_core),
                    "is_bridge_frame": bool(image_id in bridge_set),
                    "mean_confidence": frame_conf,
                    "_score": candidate_score,
                }

    records = []
    for image_id in sorted(best_records):
        record = dict(best_records[image_id])
        record.pop("_score", None)
        records.append(record)
    return records


def _transform_predictions(
    global_transforms: Dict[int, Tuple[float, np.ndarray, np.ndarray]],
    predictions_by_chunk: Dict[int, ChunkPrediction],
) -> Dict[int, ChunkPrediction]:
    transformed: Dict[int, ChunkPrediction] = {}
    for chunk_id, prediction in predictions_by_chunk.items():
        s, R, t = global_transforms.get(chunk_id, (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))
        transformed[chunk_id] = ChunkPrediction(
            chunk_id=prediction.chunk_id,
            image_ids=list(prediction.image_ids),
            image_paths=list(prediction.image_paths),
            extrinsic=apply_sim3_to_camera_poses(prediction.extrinsic, s, R, t),
            intrinsic=np.asarray(prediction.intrinsic, dtype=np.float32),
            world_points=apply_sim3_to_points(prediction.world_points, s, R, t),
            world_points_conf=np.asarray(prediction.world_points_conf, dtype=np.float32),
            images=np.asarray(prediction.images, dtype=np.float32),
            mask=None if prediction.mask is None else np.asarray(prediction.mask, dtype=bool),
            camera_poses=apply_sim3_to_camera_poses(
                prediction.camera_poses if prediction.camera_poses is not None else prediction.extrinsic,
                s,
                R,
                t,
            ),
            local_points=None if prediction.local_points is None else np.asarray(prediction.local_points, dtype=np.float32),
            conf_prob=None if prediction.conf_prob is None else np.asarray(prediction.conf_prob, dtype=np.float32),
            metadata=dict(prediction.metadata),
        )
    return transformed


def run_priority2_reconstruction(args: ReconstructionArgs) -> Dict[str, object]:
    if args.align_mode != "graph_mst":
        raise ValueError("Priority 2 currently supports align_mode=graph_mst only")

    from loop_utils.config_utils import load_config

    config = load_config(args.config_path)
    weight_key = validate_backbone_config(args.backbone, config)
    config_hash = _config_hash(args.config_path, config)
    dirs = reconstruction_dirs(args.output_dir, args.chunk_cache_dir)
    chunk_graph = load_chunk_graph_from_output(args.output_dir)
    chunk_map = {chunk.id: chunk for chunk in chunk_graph.chunks}

    predictions_by_chunk: Dict[int, ChunkPrediction] = {}
    failed_chunks: List[Dict[str, object]] = []
    cache_events: List[Dict[str, object]] = []
    cache_infos = {
        chunk.id: chunk_prediction_cache_info(chunk, args.backbone, config_hash, dirs["chunk_predictions_dir"])
        for chunk in chunk_graph.chunks
    }
    needs_inference = any(not info.cache_hit for info in cache_infos.values())

    model = None
    adapter_class = "Pi3Adapter" if args.backbone == "pi3" else "VGGTAdapter"
    if needs_inference:
        model = build_model_adapter(args.backbone, config)
        adapter_class = type(model).__name__
        model.load()

    for chunk in sorted(chunk_graph.chunks, key=lambda item: item.id):
        cache_info = cache_infos[chunk.id]
        if cache_info.cache_hit:
            prediction = load_chunk_prediction(cache_info.cache_path)
            predictions_by_chunk[chunk.id] = prediction
            cache_events.append({"chunk_id": chunk.id, "cache_hit": True, "cache_path": cache_info.cache_path})
            continue

        try:
            prediction_dict = model.infer_chunk(_chunk_image_paths(chunk))
            for key, value in list(prediction_dict.items()):
                if hasattr(value, "detach"):
                    prediction_dict[key] = value.detach().cpu().numpy().squeeze(0)
            prediction_dict["_backbone"] = args.backbone
            prediction = _to_chunk_prediction(chunk, prediction_dict, cache_info.cache_key)
            save_chunk_prediction(cache_info.cache_path, prediction)
            predictions_by_chunk[chunk.id] = prediction
            cache_events.append({"chunk_id": chunk.id, "cache_hit": False, "cache_path": cache_info.cache_path})
        except Exception as exc:  # pragma: no cover - runtime depends on checkpoint/device
            failed_chunks.append(
                {
                    "chunk_id": chunk.id,
                    "image_ids": list(chunk.image_ids),
                    "error": str(exc),
                }
            )

    if model is not None:
        del model

    alignments, _updated_edges = build_chunk_alignments(chunk_graph, predictions_by_chunk)
    chunk_pose_records, global_transforms, component_records = build_global_chunk_poses(chunk_graph, alignments)
    transformed_predictions = _transform_predictions(global_transforms, predictions_by_chunk)
    image_pose_records = _global_image_pose_records(chunk_graph, transformed_predictions)
    points, colors, confs = merge_pointclouds(
        transformed_predictions=transformed_predictions,
        chunks_by_id=chunk_map,
        pointcloud_conf_quantile=args.pointcloud_conf_quantile,
        pointcloud_sample_ratio=args.pointcloud_sample_ratio,
    )

    save_chunk_alignments(os.path.join(dirs["reconstruction_dir"], "chunk_alignments.json"), alignments)
    save_global_poses(os.path.join(dirs["reconstruction_dir"], "global_poses.json"), chunk_pose_records, image_pose_records)
    save_components(os.path.join(dirs["reconstruction_dir"], "components.json"), component_records, failed_chunks)
    save_merged_pointcloud(
        os.path.join(dirs["reconstruction_dir"], "merged_pointcloud.ply"),
        os.path.join(dirs["reconstruction_dir"], "merged_pointcloud.npz"),
        points,
        colors,
        confs,
    )

    successful_alignments = [item for item in alignments if item.sim3 is not None and item.residual is not None]
    residuals = [item.residual for item in successful_alignments if item.residual is not None]
    summary = {
        "method": f"order_free_reconstruction_mvp_v2_{args.backbone}",
        "backbone": args.backbone,
        "adapter_class": adapter_class,
        "weights_key_used": weight_key,
        "config_reference_frame_mid": bool(config.get("Model", {}).get("reference_frame_mid", False)) if args.backbone == "vggt" else None,
        "num_chunks": len(chunk_graph.chunks),
        "num_successful_chunk_predictions": len(predictions_by_chunk),
        "failed_chunks": failed_chunks,
        "cache_events": cache_events,
        "aligned_edge_count": len(successful_alignments),
        "component_count": len(component_records),
        "root_chunk_ids": [item["root_chunk_id"] for item in component_records],
        "mean_alignment_residual": float(np.mean(residuals)) if residuals else None,
        "median_alignment_residual": float(np.median(residuals)) if residuals else None,
        "num_global_image_poses": len(image_pose_records),
        "merged_point_count": int(points.shape[0]),
        "notes": [
            f"local reconstruction uses {args.backbone}",
            "chunk alignment uses shared bridge-frame camera centers",
            "global synchronization uses maximum spanning tree composition",
            "point-map alignment fallback and loop optimization are not enabled in this milestone",
        ],
    }
    write_json(os.path.join(args.output_dir, "logs", "reconstruction_summary.json"), summary)
    return summary
