from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .io_utils import descriptor_cache_key, ensure_dir, stable_hash
from .types import ImageNode


@dataclass
class DescriptorExtractionResult:
    nodes: List[ImageNode]
    extractor_name: str
    descriptor_dim: int
    fallback_used: bool
    fallback_reason: Optional[str]
    cache_dir: str
    cache_hits: int
    cache_misses: int
    events: List[Dict[str, object]]


@contextlib.contextmanager
def _pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _normalize_descriptor(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


def _fallback_descriptor(image_path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    image = Image.open(image_path).convert("RGB").resize((160, 160))
    image_np = np.asarray(image)

    if cv2 is not None:
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        edges = cv2.Canny(gray, 100, 200)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture = np.array(
            [
                float(edges.mean()) / 255.0,
                float(lap.mean()),
                float(lap.std()),
                float(gray.mean()) / 255.0,
                float(gray.std()) / 255.0,
            ],
            dtype=np.float32,
        )
        descriptor = np.concatenate([h_hist, s_hist, v_hist, gray_hist, texture], axis=0)
    else:
        rgb_hist = []
        for channel in range(3):
            hist, _ = np.histogram(image_np[..., channel], bins=16, range=(0, 255))
            rgb_hist.append(hist.astype(np.float32))
        gray = np.asarray(image.convert("L"))
        gray_hist, _ = np.histogram(gray, bins=16, range=(0, 255))
        texture = np.array(
            [
                float(gray.mean()) / 255.0,
                float(gray.std()) / 255.0,
                float(np.abs(np.diff(gray.astype(np.float32), axis=0)).mean()) / 255.0,
                float(np.abs(np.diff(gray.astype(np.float32), axis=1)).mean()) / 255.0,
            ],
            dtype=np.float32,
        )
        descriptor = np.concatenate(rgb_hist + [gray_hist.astype(np.float32), texture], axis=0)

    return _normalize_descriptor(descriptor)


def _load_cached_descriptor(cache_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(cache_path):
        return None
    try:
        return np.load(cache_path).astype(np.float32)
    except Exception:
        return None


def _save_descriptor(cache_path: str, descriptor: np.ndarray) -> None:
    ensure_dir(os.path.dirname(cache_path))
    np.save(cache_path, descriptor.astype(np.float32))


def _extract_with_fallback(
    image_items: List[Dict[str, object]],
    cache_dir: str,
    config_hash: str,
) -> DescriptorExtractionResult:
    nodes: List[ImageNode] = []
    cache_hits = 0
    cache_misses = 0
    events: List[Dict[str, object]] = []
    for item in image_items:
        image_id = int(item["original_id"])
        image_path = str(item["path"])
        key = descriptor_cache_key(image_path, "opencv_fallback", config_hash)
        cache_path = os.path.join(cache_dir, f"{key}.npy")
        descriptor = _load_cached_descriptor(cache_path)
        if descriptor is None:
            descriptor = _fallback_descriptor(image_path)
            _save_descriptor(cache_path, descriptor)
            cache_misses += 1
        else:
            cache_hits += 1
        nodes.append(
            ImageNode(
                id=image_id,
                path=image_path,
                descriptor=descriptor,
                metadata={
                    "extractor": "opencv_fallback",
                    "descriptor_dim": int(descriptor.shape[0]),
                    "cache_ref": os.path.relpath(cache_path, cache_dir),
                    "fallback_used": True,
                    "original_sorted_index": image_id,
                },
            )
        )
    events.append({"type": "fallback", "extractor": "opencv_fallback", "reason": "torch_or_model_unavailable"})
    descriptor_dim = int(nodes[0].descriptor.shape[0]) if nodes else 0
    return DescriptorExtractionResult(
        nodes=nodes,
        extractor_name="opencv_fallback",
        descriptor_dim=descriptor_dim,
        fallback_used=True,
        fallback_reason="torch_or_model_unavailable",
        cache_dir=cache_dir,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        events=events,
    )


def _extract_with_salad_dino(
    image_items: List[Dict[str, object]],
    cache_dir: str,
    config_hash: str,
    repo_root: str,
    vggt_long_config: Dict[str, object],
) -> DescriptorExtractionResult:
    import torch
    import torchvision.transforms as T

    from LoopModels.vpr_model import VPRModel

    key_name = "salad_dino"
    cached_descriptors: Dict[int, np.ndarray] = {}
    uncached: List[Tuple[int, str, str]] = []
    cache_hits = 0
    cache_misses = 0
    for item in image_items:
        image_id = int(item["original_id"])
        image_path = str(item["path"])
        key = descriptor_cache_key(image_path, key_name, config_hash)
        cache_path = os.path.join(cache_dir, f"{key}.npy")
        descriptor = _load_cached_descriptor(cache_path)
        if descriptor is not None:
            cached_descriptors[image_id] = descriptor
            cache_hits += 1
        else:
            uncached.append((image_id, image_path, cache_path))

    if uncached:
        with _pushd(repo_root):
            model = VPRModel(
                backbone_arch="dinov2_vitb14",
                backbone_config={
                    "num_trainable_blocks": 4,
                    "return_token": True,
                    "norm_layer": True,
                },
                agg_arch="SALAD",
                agg_config={
                    "num_channels": 768,
                    "num_clusters": 64,
                    "cluster_dim": 128,
                    "token_dim": 256,
                },
                vggt_long_config=vggt_long_config,
            )
            model.load_state_dict(torch.load(vggt_long_config["Weights"]["SALAD"], map_location="cpu"))
        model = model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        transform = T.Compose(
            [
                T.Resize(vggt_long_config["Loop"]["SALAD"]["image_size"], interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        batch_size = int(vggt_long_config["Loop"]["SALAD"]["batch_size"])
        autocast_enabled = device.type == "cuda"
        autocast_dtype = torch.float16
        for start in range(0, len(uncached), batch_size):
            batch_items = uncached[start : start + batch_size]
            batch = []
            for _, image_path, _ in batch_items:
                image = Image.open(image_path).convert("RGB")
                batch.append(transform(image))
            tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    desc = model(tensor).detach().cpu().numpy().astype(np.float32)
            for idx, (_, _, cache_path) in enumerate(batch_items):
                normalized = _normalize_descriptor(desc[idx])
                _save_descriptor(cache_path, normalized)
                cached_descriptors[batch_items[idx][0]] = normalized
                cache_misses += 1

    nodes: List[ImageNode] = []
    for item in image_items:
        image_id = int(item["original_id"])
        image_path = str(item["path"])
        descriptor = cached_descriptors[image_id]
        key = descriptor_cache_key(image_path, key_name, config_hash)
        cache_path = os.path.join(cache_dir, f"{key}.npy")
        nodes.append(
            ImageNode(
                id=image_id,
                path=image_path,
                descriptor=descriptor,
                metadata={
                    "extractor": key_name,
                    "descriptor_dim": int(descriptor.shape[0]),
                    "cache_ref": os.path.relpath(cache_path, cache_dir),
                    "fallback_used": False,
                    "original_sorted_index": image_id,
                },
            )
        )

    descriptor_dim = int(nodes[0].descriptor.shape[0]) if nodes else 0
    return DescriptorExtractionResult(
        nodes=nodes,
        extractor_name=key_name,
        descriptor_dim=descriptor_dim,
        fallback_used=False,
        fallback_reason=None,
        cache_dir=cache_dir,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        events=[],
    )


def _extract_with_dino_token(
    image_items: List[Dict[str, object]],
    cache_dir: str,
    config_hash: str,
    repo_root: str,
    vggt_long_config: Dict[str, object],
) -> DescriptorExtractionResult:
    import torch
    import torchvision.transforms as T

    from LoopModels.backbones.dinov2 import DINOv2

    key_name = "dino_global_token"
    cached_descriptors: Dict[int, np.ndarray] = {}
    uncached: List[Tuple[int, str, str]] = []
    cache_hits = 0
    cache_misses = 0
    for item in image_items:
        image_id = int(item["original_id"])
        image_path = str(item["path"])
        key = descriptor_cache_key(image_path, key_name, config_hash)
        cache_path = os.path.join(cache_dir, f"{key}.npy")
        descriptor = _load_cached_descriptor(cache_path)
        if descriptor is not None:
            cached_descriptors[image_id] = descriptor
            cache_hits += 1
        else:
            uncached.append((image_id, image_path, cache_path))

    if uncached:
        with _pushd(repo_root):
            model = DINOv2(
                model_name="dinov2_vitb14",
                num_trainable_blocks=4,
                norm_layer=True,
                return_token=True,
                vggt_long_config=vggt_long_config,
            )
        model = model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        transform = T.Compose(
            [
                T.Resize((336, 336), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        batch_size = 16
        for start in range(0, len(uncached), batch_size):
            batch_items = uncached[start : start + batch_size]
            batch = []
            for _, image_path, _ in batch_items:
                image = Image.open(image_path).convert("RGB")
                batch.append(transform(image))
            tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                _, tokens = model(tensor)
                desc = tokens.detach().cpu().numpy().astype(np.float32)
            for idx, (_, _, cache_path) in enumerate(batch_items):
                normalized = _normalize_descriptor(desc[idx])
                _save_descriptor(cache_path, normalized)
                cached_descriptors[batch_items[idx][0]] = normalized
                cache_misses += 1

    nodes: List[ImageNode] = []
    for item in image_items:
        image_id = int(item["original_id"])
        image_path = str(item["path"])
        descriptor = cached_descriptors[image_id]
        key = descriptor_cache_key(image_path, key_name, config_hash)
        cache_path = os.path.join(cache_dir, f"{key}.npy")
        nodes.append(
            ImageNode(
                id=image_id,
                path=image_path,
                descriptor=descriptor,
                metadata={
                    "extractor": key_name,
                    "descriptor_dim": int(descriptor.shape[0]),
                    "cache_ref": os.path.relpath(cache_path, cache_dir),
                    "fallback_used": False,
                    "original_sorted_index": image_id,
                },
            )
        )

    descriptor_dim = int(nodes[0].descriptor.shape[0]) if nodes else 0
    return DescriptorExtractionResult(
        nodes=nodes,
        extractor_name=key_name,
        descriptor_dim=descriptor_dim,
        fallback_used=False,
        fallback_reason=None,
        cache_dir=cache_dir,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        events=[],
    )


def extract_image_descriptors(
    image_items: List[Dict[str, object]],
    output_dir: str,
    repo_root: str,
    vggt_long_config: Dict[str, object],
) -> DescriptorExtractionResult:
    cache_dir = ensure_dir(os.path.join(output_dir, "logs", "descriptor_cache"))
    if not image_items:
        return DescriptorExtractionResult(
            nodes=[],
            extractor_name="none",
            descriptor_dim=0,
            fallback_used=False,
            fallback_reason=None,
            cache_dir=cache_dir,
            cache_hits=0,
            cache_misses=0,
            events=[],
        )
    config_hash = stable_hash(
        {
            "weights": vggt_long_config.get("Weights", {}),
            "loop_salad": vggt_long_config.get("Loop", {}).get("SALAD", {}),
        }
    )
    events: List[Dict[str, object]] = []

    try:
        result = _extract_with_salad_dino(image_items, cache_dir, config_hash, repo_root, vggt_long_config)
        result.events.extend(events)
        return result
    except Exception as exc:
        events.append({"type": "fallback", "extractor": "salad_dino", "reason": str(exc)})

    try:
        result = _extract_with_dino_token(image_items, cache_dir, config_hash, repo_root, vggt_long_config)
        result.fallback_used = True
        result.fallback_reason = "salad_dino_failed"
        result.events = events + result.events
        return result
    except Exception as exc:
        events.append({"type": "fallback", "extractor": "dino_global_token", "reason": str(exc)})

    result = _extract_with_fallback(image_items, cache_dir, config_hash)
    result.events = events + result.events
    result.fallback_reason = events[-1]["reason"] if events else result.fallback_reason
    return result
