"""Graph export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def export_jsonl(graph: nx.MultiDiGraph, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for node_id, attrs in graph.nodes(data=True):
            record = {
                "record_type": "node",
                "id": node_id,
                **{key: _jsonable(value) for key, value in attrs.items()},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            record = {
                "record_type": "edge",
                "source_node": source,
                "target_node": target,
                "edge_key": key,
                **{attr_key: _jsonable(value) for attr_key, value in attrs.items()},
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _graphml_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(value, sort_keys=True)


def export_graphml(graph: nx.MultiDiGraph, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graphml_graph = nx.MultiDiGraph()

    for node_id, attrs in graph.nodes(data=True):
        graphml_graph.add_node(
            node_id, **{key: _graphml_value(value) for key, value in attrs.items()}
        )
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        graphml_graph.add_edge(
            source,
            target,
            key=key,
            **{attr_key: _graphml_value(value) for attr_key, value in attrs.items()},
        )

    nx.write_graphml(graphml_graph, output_path)
