from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .types import ChunkAlignment, ChunkEdge, ChunkGraph, ChunkPrediction, GlobalChunkPose


def camera_centers_from_poses(c2w_poses: np.ndarray) -> np.ndarray:
    return np.asarray(c2w_poses)[..., :3, 3]


def sim3_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = (float(s) * np.asarray(R, dtype=np.float32))
    matrix[:3, 3] = np.asarray(t, dtype=np.float32)
    return matrix


def decompose_sim3(matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    scaled_R = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(scaled_R)))
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        raise ValueError("Degenerate Sim(3) matrix")
    R = scaled_R / scale
    return scale, R.astype(np.float32), matrix[:3, 3].astype(np.float32)


def compose_sim3(first: Tuple[float, np.ndarray, np.ndarray], second: Tuple[float, np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray]:
    s1, R1, t1 = first
    s2, R2, t2 = second
    s = float(s1) * float(s2)
    R = np.asarray(R1) @ np.asarray(R2)
    t = float(s1) * (np.asarray(R1) @ np.asarray(t2)) + np.asarray(t1)
    return float(s), R.astype(np.float32), t.astype(np.float32)


def invert_sim3(transform: Tuple[float, np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray]:
    s, R, t = transform
    if abs(float(s)) < 1e-8:
        raise ValueError("Degenerate scale in Sim(3) inverse")
    inv_s = 1.0 / float(s)
    inv_R = np.asarray(R).T
    inv_t = -(inv_s * (inv_R @ np.asarray(t)))
    return float(inv_s), inv_R.astype(np.float32), inv_t.astype(np.float32)


def apply_sim3_to_points(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    rotated = np.einsum("ij,...j->...i", np.asarray(R, dtype=np.float32), np.asarray(points, dtype=np.float32))
    return (float(s) * rotated + np.asarray(t, dtype=np.float32)).astype(np.float32)


def apply_sim3_to_camera_pose(c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = float(s) * np.asarray(R, dtype=np.float32)
    transform[:3, 3] = np.asarray(t, dtype=np.float32)
    transformed = transform @ np.asarray(c2w, dtype=np.float32)
    transformed[:3, :3] /= float(s)
    return transformed.astype(np.float32)


def apply_sim3_to_camera_poses(c2w_poses: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    transformed = np.empty_like(np.asarray(c2w_poses, dtype=np.float32))
    for idx, pose in enumerate(c2w_poses):
        transformed[idx] = apply_sim3_to_camera_pose(pose, s, R, t)
    return transformed


def estimate_sim3_umeyama(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(source_points, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source_points and target_points must be shaped [N, 3]")
    if src.shape[0] < 2:
        raise ValueError("At least two points are required for Sim(3) estimation")

    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    src_centered = src - mu_src
    tgt_centered = tgt - mu_tgt

    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    if not np.isfinite(src_var) or src_var < 1e-12:
        raise ValueError("Degenerate source point set")

    covariance = (tgt_centered.T @ src_centered) / src.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(singular_values) @ S) / src_var)
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        raise ValueError("Degenerate scale from Umeyama")
    t = mu_tgt - scale * (R @ mu_src)
    return scale, R.astype(np.float32), t.astype(np.float32)


def alignment_residual(source_points: np.ndarray, target_points: np.ndarray, transform: Tuple[float, np.ndarray, np.ndarray]) -> float:
    s, R, t = transform
    aligned = apply_sim3_to_points(source_points, s, R, t)
    residuals = np.linalg.norm(aligned - np.asarray(target_points, dtype=np.float32), axis=1)
    return float(np.mean(residuals))


def build_chunk_alignments(chunk_graph: ChunkGraph, predictions_by_chunk: Dict[int, ChunkPrediction]) -> Tuple[List[ChunkAlignment], List[ChunkEdge]]:
    chunk_map = {chunk.id: chunk for chunk in chunk_graph.chunks}
    alignments: List[ChunkAlignment] = []
    updated_edges: List[ChunkEdge] = []

    for edge in chunk_graph.edges:
        pred_a = predictions_by_chunk.get(edge.chunk_a)
        pred_b = predictions_by_chunk.get(edge.chunk_b)
        shared_ids = list(edge.shared_image_ids)
        metadata = dict(edge.metadata)
        metadata.setdefault("alignment_method", "camera_centers_umeyama")

        if pred_a is None or pred_b is None:
            metadata["reliability"] = "missing_chunk_prediction"
            alignments.append(
                ChunkAlignment(
                    chunk_a=edge.chunk_a,
                    chunk_b=edge.chunk_b,
                    shared_image_ids=shared_ids,
                    sim3=None,
                    residual=None,
                    shared_camera_centers_a=None,
                    shared_camera_centers_b=None,
                    weight=0.0,
                    reliable=False,
                    metadata=metadata,
                )
            )
            updated_edges.append(
                ChunkEdge(edge.chunk_a, edge.chunk_b, shared_ids, 0.0, None, None, metadata=metadata)
            )
            continue

        idx_map_a = {image_id: idx for idx, image_id in enumerate(pred_a.image_ids)}
        idx_map_b = {image_id: idx for idx, image_id in enumerate(pred_b.image_ids)}
        shared_ids = [image_id for image_id in shared_ids if image_id in idx_map_a and image_id in idx_map_b]
        metadata["shared_count"] = len(shared_ids)

        if len(shared_ids) < 2:
            metadata["reliability"] = "unreliable_no_alignment"
            alignments.append(
                ChunkAlignment(
                    chunk_a=edge.chunk_a,
                    chunk_b=edge.chunk_b,
                    shared_image_ids=shared_ids,
                    sim3=None,
                    residual=None,
                    shared_camera_centers_a=None,
                    shared_camera_centers_b=None,
                    weight=0.0,
                    reliable=False,
                    metadata=metadata,
                )
            )
            updated_edges.append(
                ChunkEdge(edge.chunk_a, edge.chunk_b, shared_ids, 0.0, None, None, metadata=metadata)
            )
            continue

        centers_a = camera_centers_from_poses(pred_a.camera_poses if pred_a.camera_poses is not None else pred_a.extrinsic)
        centers_b = camera_centers_from_poses(pred_b.camera_poses if pred_b.camera_poses is not None else pred_b.extrinsic)
        shared_centers_a = np.stack([centers_a[idx_map_a[image_id]] for image_id in shared_ids], axis=0)
        shared_centers_b = np.stack([centers_b[idx_map_b[image_id]] for image_id in shared_ids], axis=0)

        try:
            transform = estimate_sim3_umeyama(shared_centers_b, shared_centers_a)
            residual = alignment_residual(shared_centers_b, shared_centers_a, transform)
            reliability = "normal" if len(shared_ids) >= 3 else "weak_geometry"
            base_weight = float(edge.weight)
            weight = float(base_weight * len(shared_ids) / (1.0 + residual))
            s, R, t = transform
            matrix = sim3_matrix(s, R, t)
            metadata.update(
                {
                    "reliability": reliability,
                    "alignment_method": "camera_centers_umeyama",
                    "scale": float(s),
                    "rotation": R.tolist(),
                    "translation": t.tolist(),
                }
            )
            reliable = True
        except Exception as exc:  # pragma: no cover - failure path is dataset-dependent
            matrix = None
            residual = None
            weight = 0.0
            reliable = False
            metadata["reliability"] = "failed_alignment"
            metadata["alignment_error"] = str(exc)

        alignments.append(
            ChunkAlignment(
                chunk_a=edge.chunk_a,
                chunk_b=edge.chunk_b,
                shared_image_ids=shared_ids,
                sim3=matrix,
                residual=residual,
                shared_camera_centers_a=shared_centers_a,
                shared_camera_centers_b=shared_centers_b,
                weight=weight,
                reliable=reliable,
                metadata=metadata,
            )
        )
        updated_edges.append(
            ChunkEdge(
                chunk_a=edge.chunk_a,
                chunk_b=edge.chunk_b,
                shared_image_ids=shared_ids,
                weight=weight,
                sim3=matrix,
                residual=residual,
                metadata=metadata,
            )
        )

    chunk_graph.edges = updated_edges
    return alignments, updated_edges


def _connected_components(chunk_ids: Iterable[int], edges: List[ChunkEdge]) -> List[List[int]]:
    adjacency: Dict[int, List[int]] = {chunk_id: [] for chunk_id in chunk_ids}
    for edge in edges:
        if edge.sim3 is None:
            continue
        adjacency.setdefault(edge.chunk_a, []).append(edge.chunk_b)
        adjacency.setdefault(edge.chunk_b, []).append(edge.chunk_a)
    visited = set()
    components: List[List[int]] = []
    for chunk_id in sorted(adjacency):
        if chunk_id in visited:
            continue
        queue = deque([chunk_id])
        visited.add(chunk_id)
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _maximum_spanning_tree(component: List[int], edges: List[ChunkEdge]) -> List[ChunkEdge]:
    parent = {node: node for node in component}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    candidate_edges = [
        edge
        for edge in edges
        if edge.sim3 is not None and edge.chunk_a in parent and edge.chunk_b in parent
    ]
    ordered = sorted(candidate_edges, key=lambda edge: (-edge.weight, edge.chunk_a, edge.chunk_b))
    mst = []
    for edge in ordered:
        if union(edge.chunk_a, edge.chunk_b):
            mst.append(edge)
    return mst


def build_global_chunk_poses(chunk_graph: ChunkGraph, alignments: List[ChunkAlignment]) -> Tuple[List[GlobalChunkPose], Dict[int, Tuple[float, np.ndarray, np.ndarray]], List[Dict[str, object]]]:
    chunk_ids = [chunk.id for chunk in chunk_graph.chunks]
    reliable_edges = [edge for edge in chunk_graph.edges if edge.sim3 is not None and edge.residual is not None]
    components = _connected_components(chunk_ids, reliable_edges)

    global_transforms: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
    global_pose_records: List[GlobalChunkPose] = []
    component_records: List[Dict[str, object]] = []

    adjacency: Dict[int, List[Tuple[int, Tuple[float, np.ndarray, np.ndarray], ChunkEdge]]] = defaultdict(list)
    for edge in reliable_edges:
        forward = decompose_sim3(edge.sim3)
        backward = invert_sim3(forward)
        adjacency[edge.chunk_a].append((edge.chunk_b, forward, edge))
        adjacency[edge.chunk_b].append((edge.chunk_a, backward, edge))

    for component_id, component in enumerate(components):
        component_edges = [edge for edge in reliable_edges if edge.chunk_a in component and edge.chunk_b in component]
        mst_edges = _maximum_spanning_tree(component, component_edges)
        degree_weights = defaultdict(float)
        for edge in component_edges:
            degree_weights[edge.chunk_a] += edge.weight
            degree_weights[edge.chunk_b] += edge.weight
        root = max(sorted(component), key=lambda chunk_id: (degree_weights.get(chunk_id, 0.0), -chunk_id))

        mst_adjacency: Dict[int, List[Tuple[int, Tuple[float, np.ndarray, np.ndarray], ChunkEdge]]] = defaultdict(list)
        for edge in mst_edges:
            forward = decompose_sim3(edge.sim3)
            backward = invert_sim3(forward)
            mst_adjacency[edge.chunk_a].append((edge.chunk_b, forward, edge))
            mst_adjacency[edge.chunk_b].append((edge.chunk_a, backward, edge))
            edge.metadata["rooted_in_mst"] = True

        queue = deque([root])
        global_transforms[root] = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        visited = {root}
        while queue:
            node = queue.popleft()
            parent_transform = global_transforms[node]
            for neighbor, edge_transform, _edge in mst_adjacency.get(node, []):
                if neighbor in visited:
                    continue
                global_transforms[neighbor] = compose_sim3(parent_transform, edge_transform)
                visited.add(neighbor)
                queue.append(neighbor)

        for chunk_id in component:
            if chunk_id not in global_transforms:
                global_transforms[chunk_id] = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            s, R, t = global_transforms[chunk_id]
            global_pose_records.append(
                GlobalChunkPose(
                    chunk_id=chunk_id,
                    sim3_to_root=sim3_matrix(s, R, t),
                    component_id=component_id,
                    is_root=(chunk_id == root),
                    metadata={
                        "scale": float(s),
                        "translation": np.asarray(t, dtype=np.float32).tolist(),
                    },
                )
            )

        component_records.append(
            {
                "component_id": component_id,
                "chunk_ids": component,
                "root_chunk_id": root,
                "edge_count": len(component_edges),
                "mst_edge_count": len(mst_edges),
            }
        )

    return sorted(global_pose_records, key=lambda item: item.chunk_id), global_transforms, component_records
