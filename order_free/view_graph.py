from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .types import ImageNode, ViewEdge


def score_edge_geometry(*args, **kwargs):
    """Reserved for V2 geometry verification."""
    return None


def merge_app_geom_scores(app_score: float, geom_score: float | None) -> float:
    if geom_score is None:
        return float(app_score)
    return float(0.5 * app_score + 0.5 * geom_score)


def _descriptor_matrix(nodes: List[ImageNode]) -> np.ndarray:
    descriptors = []
    for node in nodes:
        if node.descriptor is None:
            raise ValueError(f"Image node {node.id} is missing a descriptor")
        descriptors.append(node.descriptor.astype(np.float32, copy=False))
    matrix = np.stack(descriptors, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_view_graph(nodes: List[ImageNode], knn: int = 10, mutual_knn: bool = True) -> Tuple[List[ViewEdge], Dict[str, object]]:
    if not nodes:
        return [], {"candidate_edge_count": 0, "kept_edge_count": 0}

    matrix = _descriptor_matrix(nodes)
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    num_nodes = similarity.shape[0]
    candidate_neighbors: Dict[int, List[int]] = {}
    rank_lookup: Dict[Tuple[int, int], int] = {}

    for i in range(num_nodes):
        order = np.argsort(-similarity[i])
        order = [idx for idx in order.tolist() if idx != i][: min(knn, num_nodes - 1)]
        candidate_neighbors[i] = order
        for rank, j in enumerate(order, start=1):
            rank_lookup[(i, j)] = rank

    edge_map: Dict[Tuple[int, int], ViewEdge] = {}
    for i in range(num_nodes):
        for j in candidate_neighbors[i]:
            is_mutual = i in candidate_neighbors.get(j, [])
            if mutual_knn and not is_mutual:
                continue
            key = (i, j) if i < j else (j, i)
            app_score = float((similarity[i, j] + 1.0) / 2.0)
            edge = ViewEdge(
                i=key[0],
                j=key[1],
                app_score=app_score,
                geom_score=None,
                weight=merge_app_geom_scores(app_score, None),
                metadata={
                    "rank_i_to_j": rank_lookup.get((i, j)),
                    "rank_j_to_i": rank_lookup.get((j, i)),
                    "mutual_knn": bool(is_mutual),
                    "normalized_app_score": app_score,
                    "fallback_used": bool(nodes[i].metadata.get("fallback_used") or nodes[j].metadata.get("fallback_used")),
                    "extractor": nodes[i].metadata.get("extractor"),
                },
            )
            if key not in edge_map or edge.weight > edge_map[key].weight:
                edge_map[key] = edge

    edges = sorted(edge_map.values(), key=lambda edge: (-edge.weight, edge.i, edge.j))
    stats = {
        "candidate_edge_count": int(sum(len(v) for v in candidate_neighbors.values())),
        "kept_edge_count": len(edges),
        "knn": knn,
        "mutual_knn": mutual_knn,
    }
    return edges, stats


def build_adjacency(num_nodes: int, edges: List[ViewEdge], weight_threshold: float | None = None) -> Dict[int, Dict[int, float]]:
    adjacency: Dict[int, Dict[int, float]] = {idx: {} for idx in range(num_nodes)}
    for edge in edges:
        if weight_threshold is not None and edge.weight < weight_threshold:
            continue
        adjacency[edge.i][edge.j] = edge.weight
        adjacency[edge.j][edge.i] = edge.weight
    return adjacency
