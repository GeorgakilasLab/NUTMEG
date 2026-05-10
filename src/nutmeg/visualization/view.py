from __future__ import annotations

import html
import networkx as nx
import ipycytoscape


def _set_info_panel(info_panel, text, font_family="sans-serif"):
    info_panel.value = (
        f"<pre style='margin:0; white-space:pre-wrap; font-family:{font_family};'>"
        + html.escape(text)
        + "</pre>"
    )


def _extract_data(payload):
    if payload is None:
        return {}
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    return {}


def format_selection_summary():
    return "Click a node or edge to inspect details."


def format_node_info(payload):
    data = _extract_data(payload)
    if not data:
        return "Click a node or edge to inspect details."

    label = data.get("full_label", data.get("label", data.get("id", "")))
    role = data.get("node_role", "")
    node_type = data.get("node_type", "")
    n_sources = int(data.get("n_sources", 0) or 0)
    n_targets = int(data.get("n_targets", 0) or 0)

    lines = [
        f"Label: {label}",
        f"Role: {role}",
    ]

    if str(node_type) == "CRE":
        lines.append(f"Type: {data.get('peak_type', '')}")
        lines.append(f"Number of targets: {n_targets}")
        lines.append(f"Number of sources: {n_sources}")
    elif str(node_type) == "TF":
        lines.append(f"Number of targets: {n_targets}")
    elif str(node_type) == "Gene":
        lines.append(f"Number of sources: {n_sources}")

    return "\n".join(lines)


def format_edge_info(payload):
    data = _extract_data(payload)
    if not data:
        return "Click a node or edge to inspect details."

    edge_type_raw = str(data.get("edge_type", ""))
    edge_type_map = {
        "TF_CRE": "TF→CRE",
        "CRE_GENE": "CRE→Gene",
    }
    edge_type = edge_type_map.get(edge_type_raw, edge_type_raw)

    attention = data.get("mean_attention_score", "")
    try:
        attention = f"{float(attention):.6f}"
    except Exception:
        pass

    lines = [
        f"Edge type: {edge_type}",
        f"Mean attention score: {attention}",
    ]
    return "\n".join(lines)


def build_stylesheet(style_config: dict):
    text = style_config["text"]

    return [
        {
            "selector": "node",
            "css": {
                "content": "data(label)",
                "background-color": "data(fill_color)",
                "border-color": "data(border_color)",
                "border-width": "data(border_width)",
                "width": "data(size)",
                "height": "data(size)",
                "shape": "data(shape)",
                "font-size": f"{text['node_font_size']}px",
                "text-valign": text["text_valign"],
                "text-halign": text["text_halign"],
                "color": text["node_text_color"],
                "text-wrap": text["text_wrap"],
                "text-max-width": f"{text['text_max_width']}px",
            },
        },
        {
            "selector": 'node[node_type = "CRE"]',
            "css": {
                "content": "",
            },
        },
        {
            "selector": "edge",
            "css": {
                "curve-style": style_config["edges"]["curve_style"],
                "target-arrow-shape": style_config["edges"]["target_arrow_shape"],
                "target-arrow-color": "data(line_color)",
                "line-color": "data(line_color)",
                "width": "data(width)",
                "opacity": style_config["edges"]["opacity"],
            },
        },
    ]


def make_graph_widget(nodes, edges, style_config: dict, layout_name="cose", info_panel=None):
    graph_nx = nx.DiGraph()

    for _, row in nodes.iterrows():
        graph_nx.add_node(
            str(row["id"]),
            id=str(row["id"]),
            label=str(row["display_label"]),
            full_label=str(row["label"]),
            node_type=str(row["node_type"]),
            node_role=str(row["node_role"]),
            peak_type=str(row.get("peak_type", "unknown")),
            n_sources=int(row.get("n_sources", 0)),
            n_targets=int(row.get("n_targets", 0)),
            total_support=float(row.get("total_support", 0.0)),
            size=float(row["node_size"]),
            fill_color=str(row["node_fill_color"]),
            border_color=str(row["node_border_color"]),
            border_width=float(row["node_border_width"]),
            shape=str(row["node_shape"]),
        )

    for _, row in edges.iterrows():
        graph_nx.add_edge(
            str(row["source"]),
            str(row["target"]),
            edge_type=str(row["edge_type"]),
            mean_attention_score=float(row["mean_attention_score"]),
            line_color=str(row["edge_color"]),
            width=float(row["edge_width"]),
        )

    graph = ipycytoscape.CytoscapeWidget()
    graph.graph.add_graph_from_networkx(graph_nx)
    graph.set_style(build_stylesheet(style_config))

    if info_panel is not None:
        font_family = style_config["panels"].get("font_family", "sans-serif")

        def _handle_node_click(payload):
            _set_info_panel(info_panel, format_node_info(payload), font_family)

        def _handle_edge_click(payload):
            _set_info_panel(info_panel, format_edge_info(payload), font_family)

        graph.on("node", "click", _handle_node_click)
        graph.on("edge", "click", _handle_edge_click)

    graph.set_layout(name=layout_name)
    graph.layout.width = style_config["layout"]["widget_width"]
    graph.layout.height = style_config["layout"]["widget_height"]

    return graph