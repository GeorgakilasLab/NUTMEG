from __future__ import annotations

import html
import ipywidgets as widgets


def _circle(fill, border="#111111", size=18, border_width=2):
    return widgets.HTML(value=f"""
        <div style="width:{size}px;height:{size}px;border-radius:50%;background:{fill};
            border:{border_width}px solid {border};display:inline-block;
            vertical-align:middle;margin-right:8px;box-sizing:border-box;"></div>""")


def _diamond(fill, border="#111111", size=18, border_width=1):
    return widgets.HTML(value=f"""
        <div style="width:{size}px;height:{size}px;background:{fill};
            border:{border_width}px solid {border};transform:rotate(45deg);
            display:inline-block;vertical-align:middle;margin-right:10px;
            box-sizing:border-box;"></div>""")


def _line(color, width=4, line_width_px=34):
    return widgets.HTML(value=f"""
        <div style="width:{line_width_px}px;height:0;
            border-top:{width}px solid {color};display:inline-block;
            vertical-align:middle;margin-right:8px;box-sizing:border-box;"></div>""")


def _row(icon_widget, text):
    return widgets.HBox(
        [icon_widget, widgets.HTML(value=f"<span>{html.escape(text)}</span>")],
        layout=widgets.Layout(align_items="center", margin="0 0 6px 0"),
    )


def make_legend_panel(style_config, rna_only: bool = False):
    ns           = style_config["node_styles"]
    es           = style_config["edge_styles"]
    legend_sizes = style_config["legend_sizes"]
    defaults     = style_config["defaults"]

    node_icon  = legend_sizes.get("node_icon", 18)
    node_small = legend_sizes.get("node_small", 12)
    node_large = legend_sizes.get("node_large", 24)
    edge_width = legend_sizes.get("edge_width", 4)
    edge_small = legend_sizes.get("edge_small", 2)
    edge_large = legend_sizes.get("edge_large", 5)
    edge_length = legend_sizes.get("edge_length", 34)
    dual_bw    = defaults.get("dual_role_border_width", 4)

    children = [
        widgets.HTML("<div style='font-weight:bold; margin:0 0 8px 0;'>Node roles</div>"),

        _row(_circle(ns["TF"]["color"], ns["TF"].get("border_color", "#111111"),
                     node_icon, ns["TF"].get("border_width", 1)),
             ns["TF"].get("label", "TF")),

        _row(_circle(ns["Gene"]["color"], ns["Gene"].get("border_color", "#111111"),
                     node_icon, ns["Gene"].get("border_width", 1)),
             ns["Gene"].get("label", "Gene")),

        _row(_circle(ns["TF"]["color"], ns["Gene"]["color"], node_icon, dual_bw),
             "TF | Gene (dual role)"),
    ]

    # CRE node types — multiome only
    if not rna_only:
        children += [
            _row(_diamond(ns["CRE_promoter"]["color"],
                          ns["CRE_promoter"].get("border_color", "#111111"),
                          node_icon, ns["CRE_promoter"].get("border_width", 1)),
                 ns["CRE_promoter"].get("label", "CRE promoter")),

            _row(_diamond(ns["CRE_distal"]["color"],
                          ns["CRE_distal"].get("border_color", "#111111"),
                          node_icon, ns["CRE_distal"].get("border_width", 1)),
                 ns["CRE_distal"].get("label", "CRE distal")),

            _row(_diamond(ns["CRE_unknown"]["color"],
                          ns["CRE_unknown"].get("border_color", "#111111"),
                          node_icon, ns["CRE_unknown"].get("border_width", 1)),
                 ns["CRE_unknown"].get("label", "CRE unknown")),
        ]

    children += [
        widgets.HTML("<div style='font-weight:bold; margin:12px 0 8px 0;'>Node size</div>"),
        _row(_circle("#999999", "#999999", node_small, 0), "Lower total support"),
        _row(_circle("#999999", "#999999", node_large, 0), "Higher total support"),
        widgets.HTML("<div style='font-weight:bold; margin:12px 0 8px 0;'>Edges</div>"),
    ]

    if rna_only:
        tf_gene_color = es.get("TF_GENE", {}).get("color", defaults["fallback_edge_color"])
        children.append(
            _row(_line(tf_gene_color, edge_width, edge_length),
                 es.get("TF_GENE", {}).get("label", "TF → Gene"))
        )
    else:
        children += [
            _row(_line(es["TF_CRE"]["color"],  edge_width, edge_length),
                 es["TF_CRE"].get("label",  "TF → CRE")),
            _row(_line(es["CRE_GENE"]["color"], edge_width, edge_length),
                 es["CRE_GENE"].get("label", "CRE → Gene")),
        ]

    children += [
        widgets.HTML("<div style='font-weight:bold; margin:12px 0 8px 0;'>Edge size</div>"),
        _row(_line("#666666", edge_small, edge_length), "Lower total support"),
        _row(_line("#666666", edge_large, edge_length), "Higher total support"),
    ]

    return widgets.VBox(
        children,
        layout=widgets.Layout(
            border=style_config["panels"]["border"],
            padding=style_config["panels"]["padding"],
            width="100%",
            overflow_y="auto",
            min_height="0",
        ),
    )