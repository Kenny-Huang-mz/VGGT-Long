from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .types import BridgeSet, GeometryCluster, ViewEdge
from .view_graph import build_adjacency


def discover_bridge_sets(
    clusters: List[GeometryCluster],
    edges: List[ViewEdge],
    num_nodes: int,
    bridge_top_m: int,
) -> Tuple[List[BridgeSet], Dict[str, object]]:
    adjacency = build_adjacency(num_nodes, edges, weight_threshold=None)
    cluster_by_id = {cluster.id: cluster for cluster in clusters}
    node_to_cluster: Dict[int, Set[int]] = defaultdict(set)
    for cluster in clusters:
        for image_id in cluster.image_ids:
            node_to_cluster[image_id].add(cluster.id)

    cluster_pairs: Dict[Tuple[int, int], float] = defaultdict(float)
    for edge in edges:
        cluster_ids_i = node_to_cluster.get(edge.i, set())
        cluster_ids_j = node_to_cluster.get(edge.j, set())
        for cluster_i in cluster_ids_i:
            for cluster_j in cluster_ids_j:
                if cluster_i == cluster_j:
                    continue
                key = tuple(sorted((cluster_i, cluster_j)))
                cluster_pairs[key] += edge.weight

    bridge_sets: List[BridgeSet] = []
    pair_stats = []
    for (cluster_a_id, cluster_b_id), pair_weight in sorted(cluster_pairs.items()):
        cluster_a = cluster_by_id[cluster_a_id]
        cluster_b = cluster_by_id[cluster_b_id]
        cluster_a_set = set(cluster_a.image_ids)
        cluster_b_set = set(cluster_b.image_ids)

        candidate_ids = set(cluster_a.image_ids) | set(cluster_b.image_ids)
        for node_id in range(num_nodes):
            if node_id in candidate_ids:
                continue
            conn_a = sum(weight for nbr, weight in adjacency[node_id].items() if nbr in cluster_a_set)
            conn_b = sum(weight for nbr, weight in adjacency[node_id].items() if nbr in cluster_b_set)
            if conn_a > 0 and conn_b > 0:
                candidate_ids.add(node_id)

        scored_candidates = []
        for node_id in sorted(candidate_ids):
            conn_a = sum(weight for nbr, weight in adjacency[node_id].items() if nbr in cluster_a_set)
            conn_b = sum(weight for nbr, weight in adjacency[node_id].items() if nbr in cluster_b_set)
            if conn_a <= 0 or conn_b <= 0:
                continue
            bridge_score = min(conn_a, conn_b)
            if node_id in cluster_a_set:
                source = "cluster_a"
            elif node_id in cluster_b_set:
                source = "cluster_b"
            else:
                source = "external_neighbor"
            scored_candidates.append(
                {
                    "image_id": node_id,
                    "score": float(bridge_score),
                    "conn_a": float(conn_a),
                    "conn_b": float(conn_b),
                    "source": source,
                }
            )

        scored_candidates.sort(key=lambda item: (-item["score"], item["image_id"]))
        selected = scored_candidates[:bridge_top_m]
        selected_ids = [item["image_id"] for item in selected]
        reliability = "strong"
        weak_reason = None
        if len(selected_ids) < 2:
            reliability = "weak"
            weak_reason = "insufficient_bridge_candidates"
        elif selected and sum(item["score"] for item in selected) / len(selected) < 0.2:
            reliability = "weak"
            weak_reason = "low_average_bridge_score"
        aggregate_score = float(sum(item["score"] for item in selected))
        bridge_sets.append(
            BridgeSet(
                cluster_a=cluster_a_id,
                cluster_b=cluster_b_id,
                bridge_image_ids=selected_ids,
                score=aggregate_score,
                metadata={
                    "reliability": reliability,
                    "weak_reason": weak_reason,
                    "pair_weight": float(pair_weight),
                    "candidate_scores": selected,
                },
            )
        )
        pair_stats.append(
            {
                "cluster_pair": [cluster_a_id, cluster_b_id],
                "candidate_count": len(scored_candidates),
                "selected_count": len(selected_ids),
                "reliability": reliability,
            }
        )

    stats = {
        "bridge_pair_count": len(bridge_sets),
        "bridge_pairs": pair_stats,
    }
    return bridge_sets, stats
