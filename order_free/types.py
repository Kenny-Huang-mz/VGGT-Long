from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


@dataclass
class ImageNode:
    id: int
    path: str
    descriptor: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_descriptor: bool = False) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "path": self.path,
            "descriptor": _to_builtin(self.descriptor) if include_descriptor and self.descriptor is not None else None,
            "metadata": _to_builtin(self.metadata),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ImageNode":
        descriptor = payload.get("descriptor")
        if descriptor is not None:
            descriptor = np.asarray(descriptor, dtype=np.float32)
        return cls(
            id=int(payload["id"]),
            path=str(payload["path"]),
            descriptor=descriptor,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ViewEdge:
    i: int
    j: int
    app_score: float
    geom_score: Optional[float]
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "i": self.i,
            "j": self.j,
            "app_score": float(self.app_score),
            "geom_score": None if self.geom_score is None else float(self.geom_score),
            "weight": float(self.weight),
            "metadata": _to_builtin(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ViewEdge":
        return cls(
            i=int(payload["i"]),
            j=int(payload["j"]),
            app_score=float(payload["app_score"]),
            geom_score=None if payload.get("geom_score") is None else float(payload["geom_score"]),
            weight=float(payload["weight"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class GeometryCluster:
    id: int
    image_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_ids": list(self.image_ids),
            "metadata": _to_builtin(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeometryCluster":
        return cls(
            id=int(payload["id"]),
            image_ids=[int(v) for v in payload["image_ids"]],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class BridgeSet:
    cluster_a: int
    cluster_b: int
    bridge_image_ids: List[int]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_a": self.cluster_a,
            "cluster_b": self.cluster_b,
            "bridge_image_ids": list(self.bridge_image_ids),
            "score": float(self.score),
            "metadata": _to_builtin(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BridgeSet":
        return cls(
            cluster_a=int(payload["cluster_a"]),
            cluster_b=int(payload["cluster_b"]),
            bridge_image_ids=[int(v) for v in payload["bridge_image_ids"]],
            score=float(payload["score"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class Chunk:
    id: int
    core_image_ids: List[int]
    bridge_image_ids: List[int]
    image_ids: List[int]
    source_cluster_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "core_image_ids": list(self.core_image_ids),
            "bridge_image_ids": list(self.bridge_image_ids),
            "image_ids": list(self.image_ids),
            "source_cluster_ids": list(self.source_cluster_ids),
            "metadata": _to_builtin(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Chunk":
        return cls(
            id=int(payload["id"]),
            core_image_ids=[int(v) for v in payload["core_image_ids"]],
            bridge_image_ids=[int(v) for v in payload["bridge_image_ids"]],
            image_ids=[int(v) for v in payload["image_ids"]],
            source_cluster_ids=[int(v) for v in payload["source_cluster_ids"]],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ChunkEdge:
    chunk_a: int
    chunk_b: int
    shared_image_ids: List[int]
    weight: float
    sim3: Optional[np.ndarray] = None
    residual: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_a": self.chunk_a,
            "chunk_b": self.chunk_b,
            "shared_image_ids": list(self.shared_image_ids),
            "weight": float(self.weight),
            "sim3": _to_builtin(self.sim3) if self.sim3 is not None else None,
            "residual": None if self.residual is None else float(self.residual),
            "metadata": _to_builtin(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkEdge":
        sim3 = payload.get("sim3")
        if sim3 is not None:
            sim3 = np.asarray(sim3, dtype=np.float32)
        return cls(
            chunk_a=int(payload["chunk_a"]),
            chunk_b=int(payload["chunk_b"]),
            shared_image_ids=[int(v) for v in payload["shared_image_ids"]],
            weight=float(payload["weight"]),
            sim3=sim3,
            residual=None if payload.get("residual") is None else float(payload["residual"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ChunkGraph:
    chunks: List[Chunk]
    edges: List[ChunkEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": _to_builtin(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkGraph":
        return cls(
            chunks=[Chunk.from_dict(item) for item in payload.get("chunks", [])],
            edges=[ChunkEdge.from_dict(item) for item in payload.get("edges", [])],
            metadata=dict(payload.get("metadata", {})),
        )
