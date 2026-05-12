from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .types import BridgeSet, Chunk, ChunkEdge, ChunkGraph, GeometryCluster


def _rank_core_candidates(
    core_ids: List[int],
    bridge_ids: List[int],
    node_centrality: Dict[int, float],
    descriptors: Dict[int, np.ndarray],
    bridge_connectivity: Dict[int, float],
) -> List[int]:
    if not core_ids:
        return []

    score_map = {}
    for image_id in core_ids:
        score_map[image_id] = 0.7 * node_centrality.get(image_id, 0.0) + 0.3 * bridge_connectivity.get(image_id, 0.0)

    ordered = sorted(core_ids, key=lambda image_id: (-score_map[image_id], image_id))
    if len(ordered) <= 2:
        return ordered

    selected = [ordered[0]]
    remaining = ordered[1:]
    while remaining:
        best_id = None
        best_score = None
        for image_id in remaining:
            desc = descriptors.get(image_id)
            if desc is None or not selected:
                diversity = 0.0
            else:
                selected_desc = np.stack([descriptors[item] for item in selected if item in descriptors], axis=0)
                if selected_desc.size == 0:
                    diversity = 0.0
                else:
                    diversity = float(np.mean(1.0 - np.clip(selected_desc @ desc, -1.0, 1.0)))
            combined = score_map[image_id] + 0.15 * diversity
            candidate = (combined, diversity, -image_id)
            if best_score is None or candidate > best_score:
                best_score = candidate
                best_id = image_id
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


def build_bridge_aware_chunks(
    clusters: List[GeometryCluster],
    bridge_sets: List[BridgeSet],
    image_paths_by_id: Dict[int, str],
    descriptors_by_id: Dict[int, np.ndarray],
    node_centrality: Dict[int, float],
    max_chunk_size: int,
) -> Tuple[ChunkGraph, Dict[str, object]]:
    bridges_by_cluster: Dict[int, List[BridgeSet]] = defaultdict(list)
    for bridge_set in bridge_sets:
        bridges_by_cluster[bridge_set.cluster_a].append(bridge_set)
        bridges_by_cluster[bridge_set.cluster_b].append(bridge_set)

    chunks: List[Chunk] = []
    adjacency_records = []
    for cluster in clusters:
        cluster_bridges = bridges_by_cluster.get(cluster.id, [])
        bridge_ids = []
        bridge_connectivity: Dict[int, float] = defaultdict(float)
        bridge_scores_by_id: Dict[int, float] = {}
        for bridge_set in cluster_bridges:
            for candidate in bridge_set.metadata.get("candidate_scores", []):
                bridge_connectivity[candidate["image_id"]] += float(candidate["score"])
            for image_id in bridge_set.bridge_image_ids:
                bridge_ids.append(image_id)
                bridge_scores_by_id[image_id] = bridge_scores_by_id.get(image_id, 0.0) + float(bridge_set.score)
        dedup_bridge_ids = sorted(set(bridge_ids))
        dedup_core_ids = sorted(set(cluster.image_ids))
        dropped_core_ids: List[int] = []
        dropped_bridge_ids: List[int] = []

        if len(dedup_bridge_ids) > max_chunk_size:
            ordered_bridge_ids = sorted(dedup_bridge_ids, key=lambda image_id: (-bridge_scores_by_id.get(image_id, 0.0), image_id))
            dropped_bridge_ids = ordered_bridge_ids[max_chunk_size:]
            dedup_bridge_ids = sorted(ordered_bridge_ids[:max_chunk_size])
            dedup_core_ids = []
        elif len(dedup_core_ids) + len(dedup_bridge_ids) > max_chunk_size:
            ranked_core_ids = _rank_core_candidates(
                dedup_core_ids,
                dedup_bridge_ids,
                node_centrality=node_centrality,
                descriptors=descriptors_by_id,
                bridge_connectivity=bridge_connectivity,
            )
            keep_count = max(0, max_chunk_size - len(dedup_bridge_ids))
            kept_core = set(ranked_core_ids[:keep_count])
            dropped_core_ids = sorted(set(dedup_core_ids) - kept_core)
            dedup_core_ids = sorted(kept_core)

        image_ids = sorted(set(dedup_core_ids) | set(dedup_bridge_ids))
        chunk = Chunk(
            id=cluster.id,
            core_image_ids=dedup_core_ids,
            bridge_image_ids=dedup_bridge_ids,
            image_ids=image_ids,
            source_cluster_ids=[cluster.id],
            metadata={
                "image_paths": [image_paths_by_id[image_id] for image_id in image_ids],
                "dropped_core_image_ids": dropped_core_ids,
                "dropped_bridge_image_ids": dropped_bridge_ids,
                "weak_cluster": bool(cluster.metadata.get("weak_cluster", False)),
            },
        )
        chunks.append(chunk)
        adjacency_records.append(
            {
                "cluster_id": cluster.id,
                "bridge_neighbor_count": len(cluster_bridges),
                "chunk_size": len(image_ids),
                "bridge_size": len(dedup_bridge_ids),
            }
        )

    chunk_edges: List[ChunkEdge] = []
    for idx, chunk_a in enumerate(chunks):
        set_a = set(chunk_a.image_ids)
        for chunk_b in chunks[idx + 1 :]:
            shared = sorted(set_a & set(chunk_b.image_ids))
            if not shared:
                continue
            weight = float(len(shared) / max(len(chunk_a.image_ids), len(chunk_b.image_ids), 1))
            chunk_edges.append(
                ChunkEdge(
                    chunk_a=chunk_a.id,
                    chunk_b=chunk_b.id,
                    shared_image_ids=shared,
                    weight=weight,
                    sim3=None,
                    residual=None,
                    metadata={
                        "shared_count": len(shared),
                        "bridge_shared_count": len(set(shared) & (set(chunk_a.bridge_image_ids) | set(chunk_b.bridge_image_ids))),
                    },
                )
            )

    chunk_graph = ChunkGraph(
        chunks=chunks,
        edges=sorted(chunk_edges, key=lambda edge: (-edge.weight, edge.chunk_a, edge.chunk_b)),
        metadata={
            "cluster_adjacency": [bridge_set.to_dict() for bridge_set in bridge_sets],
        },
    )
    stats = {
        "chunk_count": len(chunks),
        "chunk_sizes": [len(chunk.image_ids) for chunk in chunks],
        "chunk_edge_count": len(chunk_edges),
        "chunk_adjacency": adjacency_records,
        "shared_frame_counts": [len(edge.shared_image_ids) for edge in chunk_edges],
    }
    return chunk_graph, stats
