from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .io_utils import ensure_dir, stable_hash, write_json
from .types import Chunk, ChunkAlignment, ChunkGraph, ChunkPrediction, GlobalChunkPose


@dataclass
class ChunkCacheInfo:
    cache_key: str
    cache_path: str
    cache_hit: bool


def reconstruction_dirs(output_dir: str, chunk_cache_dir: str | None = None) -> Dict[str, str]:
    reconstruction_dir = ensure_dir(os.path.join(output_dir, "reconstruction"))
    chunk_predictions_dir = ensure_dir(chunk_cache_dir or os.path.join(reconstruction_dir, "chunk_predictions"))
    return {
        "reconstruction_dir": reconstruction_dir,
        "chunk_predictions_dir": chunk_predictions_dir,
    }


def chunk_cache_key(chunk: Chunk, backbone: str, config_hash: str) -> str:
    payload = {
        "chunk_id": chunk.id,
        "image_ids": sorted(chunk.image_ids),
        "backbone": backbone,
        "config_hash": config_hash,
    }
    return stable_hash(payload)


def chunk_prediction_cache_info(chunk: Chunk, backbone: str, config_hash: str, chunk_predictions_dir: str) -> ChunkCacheInfo:
    cache_key = chunk_cache_key(chunk, backbone, config_hash)
    cache_path = os.path.join(chunk_predictions_dir, f"chunk_{chunk.id}.npy")
    cache_hit = False
    if os.path.exists(cache_path):
        try:
            payload = np.load(cache_path, allow_pickle=True).item()
            cache_hit = payload.get("metadata", {}).get("cache_key") == cache_key
        except Exception:
            cache_hit = False
    return ChunkCacheInfo(cache_key=cache_key, cache_path=cache_path, cache_hit=cache_hit)


def save_chunk_prediction(cache_path: str, prediction: ChunkPrediction) -> None:
    ensure_dir(os.path.dirname(cache_path))
    np.save(cache_path, prediction.to_dict(), allow_pickle=True)


def load_chunk_prediction(cache_path: str) -> ChunkPrediction:
    payload = np.load(cache_path, allow_pickle=True).item()
    return ChunkPrediction.from_dict(payload)


def load_chunk_graph_from_output(output_dir: str) -> ChunkGraph:
    import json

    path = os.path.join(output_dir, "chunk_graph.json")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ChunkGraph.from_dict(payload["chunk_graph"])


def save_chunk_alignments(path: str, alignments: List[ChunkAlignment]) -> None:
    write_json(path, {"alignments": [alignment.to_dict() for alignment in alignments]})


def save_global_poses(path: str, chunk_poses: List[GlobalChunkPose], image_pose_records: List[Dict[str, object]]) -> None:
    write_json(
        path,
        {
            "chunk_global_poses": [item.to_dict() for item in chunk_poses],
            "image_global_poses": image_pose_records,
        },
    )


def save_components(path: str, components: List[Dict[str, object]], failed_chunks: List[Dict[str, object]]) -> None:
    write_json(path, {"components": components, "failed_chunks": failed_chunks})


def _sample_indices(size: int, ratio: float) -> np.ndarray:
    if ratio >= 1.0 or size <= 0:
        return np.arange(size)
    sample_size = max(1, int(size * ratio))
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(size, size=sample_size, replace=False))


def merge_pointclouds(
    transformed_predictions: Dict[int, ChunkPrediction],
    chunks_by_id: Dict[int, Chunk],
    pointcloud_conf_quantile: float,
    pointcloud_sample_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    all_confs: List[np.ndarray] = []

    for chunk_id in sorted(transformed_predictions):
        prediction = transformed_predictions[chunk_id]
        world_points = np.asarray(prediction.world_points, dtype=np.float32)
        conf = np.asarray(prediction.world_points_conf, dtype=np.float32)
        images = np.asarray(prediction.images, dtype=np.float32)
        colors = np.transpose(images, (0, 2, 3, 1))

        threshold = float(np.quantile(conf.reshape(-1), pointcloud_conf_quantile))
        mask = conf >= threshold
        if prediction.mask is not None:
            mask = np.logical_and(mask, np.asarray(prediction.mask, dtype=bool))

        points_flat = world_points[mask]
        colors_flat = np.clip(colors[mask] * 255.0, 0, 255).astype(np.uint8)
        conf_flat = conf[mask]
        if points_flat.size == 0:
            continue

        keep = _sample_indices(points_flat.shape[0], pointcloud_sample_ratio)
        all_points.append(points_flat[keep])
        all_colors.append(colors_flat[keep])
        all_confs.append(conf_flat[keep])

    if not all_points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.concatenate(all_points, axis=0).astype(np.float32),
        np.concatenate(all_colors, axis=0).astype(np.uint8),
        np.concatenate(all_confs, axis=0).astype(np.float32),
    )


def save_merged_pointcloud(ply_path: str, npz_path: str, points: np.ndarray, colors: np.ndarray, confs: np.ndarray) -> None:
    ensure_dir(os.path.dirname(ply_path))
    ensure_dir(os.path.dirname(npz_path))
    if points.shape[0] > 0:
        _write_ply(ply_path, points, colors)
    else:
        with open(ply_path, "w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    np.savez_compressed(npz_path, points=points, colors=colors, confs=confs)


def _write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{float(point[0])} {float(point[1])} {float(point[2])} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
