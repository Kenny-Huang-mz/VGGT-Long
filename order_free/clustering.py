from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from .types import GeometryCluster, ViewEdge
from .view_graph import build_adjacency


def _weighted_degree(adjacency: Dict[int, Dict[int, float]], nodes: Set[int]) -> Dict[int, float]:
    return {node: float(sum(weight for nbr, weight in adjacency[node].items() if nbr in nodes)) for node in nodes}


def _connected_components(adjacency: Dict[int, Dict[int, float]], node_ids: List[int]) -> List[List[int]]:
    remaining = set(node_ids)
    components: List[List[int]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component = []
        remaining.remove(seed)
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in adjacency[node]:
                if nbr in remaining:
                    remaining.remove(nbr)
                    stack.append(nbr)
        components.append(sorted(component))
    return components


def _region_grow_partition(component: List[int], adjacency: Dict[int, Dict[int, float]], max_size: int) -> List[List[int]]:
    remaining = set(component)
    degrees = _weighted_degree(adjacency, remaining)
    groups: List[List[int]] = []

    while remaining:
        seed = max(sorted(remaining), key=lambda node: (degrees.get(node, 0.0), -node))
        current = [seed]
        remaining.remove(seed)
        while remaining and len(current) < max_size:
            candidate_scores = []
            current_set = set(current)
            for candidate in remaining:
                score = sum(adjacency[candidate].get(member, 0.0) for member in current_set)
                if score > 0:
                    candidate_scores.append((score, degrees.get(candidate, 0.0), -candidate, candidate))
            if not candidate_scores:
                break
            candidate_scores.sort(reverse=True)
            next_node = candidate_scores[0][-1]
            current.append(next_node)
            remaining.remove(next_node)
        groups.append(sorted(current))
    return groups


def _merge_undersized_clusters(
    components: List[List[int]],
    adjacency_all: Dict[int, Dict[int, float]],
    min_cluster_size: int,
    max_core_size: int,
) -> Tuple[List[List[int]], List[Dict[str, object]]]:
    clusters = [list(component) for component in components]
    events: List[Dict[str, object]] = []

    changed = True
    while changed:
        changed = False
        for idx, cluster in enumerate(list(clusters)):
            if len(cluster) >= min_cluster_size:
                continue
            best_neighbor = None
            best_score = -1.0
            cluster_set = set(cluster)
            for other_idx, other in enumerate(clusters):
                if other_idx == idx:
                    continue
                other_set = set(other)
                cross_score = 0.0
                for node in cluster:
                    cross_score += sum(weight for nbr, weight in adjacency_all[node].items() if nbr in other_set)
                if cross_score > best_score and len(cluster) + len(other) <= max_core_size:
                    best_score = cross_score
                    best_neighbor = other_idx
            if best_neighbor is not None and best_score > 0:
                merged = sorted(set(clusters[idx]) | set(clusters[best_neighbor]))
                events.append(
                    {
                        "type": "merge_undersized_cluster",
                        "source_cluster_nodes": sorted(cluster),
                        "target_cluster_nodes": sorted(clusters[best_neighbor]),
                        "cross_score": best_score,
                    }
                )
                first, second = sorted([idx, best_neighbor], reverse=True)
                clusters.pop(first)
                clusters.pop(second)
                clusters.append(merged)
                changed = True
                break
    return clusters, events


def _fallback_grouping(
    num_nodes: int,
    adjacency_all: Dict[int, Dict[int, float]],
    max_core_size: int,
) -> List[List[int]]:
    remaining = set(range(num_nodes))
    degrees = _weighted_degree(adjacency_all, remaining)
    groups: List[List[int]] = []
    while remaining:
        seed = max(sorted(remaining), key=lambda node: (degrees.get(node, 0.0), -node))
        current = [seed]
        remaining.remove(seed)
        while remaining and len(current) < max_core_size:
            best = None
            best_score = -1.0
            current_set = set(current)
            for candidate in remaining:
                score = sum(adjacency_all[candidate].get(member, 0.0) for member in current_set)
                if score > best_score:
                    best_score = score
                    best = candidate
            if best is None:
                best = min(remaining)
            current.append(best)
            remaining.remove(best)
        groups.append(sorted(current))
    return groups


def build_geometry_clusters(
    num_nodes: int,
    edges: List[ViewEdge],
    min_cluster_size: int,
    max_chunk_size: int,
    bridge_top_m: int,
    weight_threshold: float | None = None,
) -> Tuple[List[GeometryCluster], Dict[str, object], List[Dict[str, object]]]:
    reserved_bridge_slots = min(max(1, max(8, bridge_top_m)), max(1, max_chunk_size - 1))
    max_core_size = max(1, max_chunk_size - reserved_bridge_slots)
    all_weights = np.asarray([edge.weight for edge in edges], dtype=np.float32) if edges else np.asarray([], dtype=np.float32)
    p40 = float(np.percentile(all_weights, 40)) if all_weights.size else 0.0
    threshold = float(weight_threshold if weight_threshold is not None else max(0.35, p40))

    adjacency_all = build_adjacency(num_nodes, edges, weight_threshold=None)
    adjacency_kept = build_adjacency(num_nodes, edges, weight_threshold=threshold)
    components = _connected_components(adjacency_kept, list(range(num_nodes)))

    fallback_events: List[Dict[str, object]] = []
    processed_components: List[List[int]] = []
    for component in components:
        if len(component) <= max_core_size:
            processed_components.append(component)
        else:
            processed_components.extend(_region_grow_partition(component, adjacency_kept, max_core_size))

    processed_components, merge_events = _merge_undersized_clusters(
        processed_components,
        adjacency_all=adjacency_all,
        min_cluster_size=min_cluster_size,
        max_core_size=max_core_size,
    )
    fallback_events.extend(merge_events)

    non_singletons = [component for component in processed_components if len(component) > 1]
    if not non_singletons and num_nodes > 0:
        fallback_groups = _fallback_grouping(num_nodes, adjacency_all, max_core_size)
        fallback_events.append(
            {
                "type": "fallback_grouping",
                "reason": "graph_too_sparse_after_threshold",
                "threshold": threshold,
                "num_nodes": num_nodes,
            }
        )
        processed_components = fallback_groups

    clusters = []
    undersized_count = 0
    for cluster_id, component in enumerate(sorted(processed_components, key=lambda item: (min(item), len(item)))):
        metadata = {"weak_cluster": len(component) < min_cluster_size}
        if len(component) < min_cluster_size:
            undersized_count += 1
        clusters.append(GeometryCluster(id=cluster_id, image_ids=sorted(component), metadata=metadata))

    stats = {
        "reserved_bridge_slots": reserved_bridge_slots,
        "max_core_size": max_core_size,
        "weight_threshold": threshold,
        "weight_threshold_source": "cli" if weight_threshold is not None else "max(0.35, p40)",
        "p40_weight": p40,
        "cluster_count": len(clusters),
        "cluster_sizes": [len(cluster.image_ids) for cluster in clusters],
        "undersized_cluster_count": undersized_count,
        "kept_edge_count": int(sum(len(neighbors) for neighbors in adjacency_kept.values()) // 2),
    }
    return clusters, stats, fallback_events
