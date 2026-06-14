from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

from .loading import (
    find_class_csvs,
    load_peak_annotations,
    load_class_table,
    filter_paths,
)
from .graph_build import build_nodes, aggregate_edges, _is_rna_only
from .legend import make_legend_panel
from .view import (
    make_graph_widget,
    _set_info_panel,
    format_selection_summary,
)
from .utils import load_json


def show_notebook_graph(
    interpretation_dir,
    peak_gene_csv=None,         
    max_q_value=0.05,
    min_cell_freq=0.05,
    min_n_cells=None,
    top_n_paths=150,
    max_tfs=None,
    class_depth=2,
    style_json=None,
):
    if style_json is None:
        style_json = Path(__file__).parent / "graph_style.json"

    style_config = load_json(style_json)

    class_to_csv = find_class_csvs(
        interpretation_dir,
        filename=None,           
        class_depth=class_depth,
    )
    classes = sorted(class_to_csv.keys())
    if not classes:
        raise ValueError(f"No classes found under {interpretation_dir}")

    _sample_df = load_class_table(class_to_csv, classes[0])
    rna_only   = _is_rna_only(_sample_df)

    if not rna_only:
        if peak_gene_csv is None:
            raise ValueError(
                "peak_gene_csv is required for multiome mode. "
                "Pass peak_gene_csv=None only for rna_only experiments."
            )
        peak_annot = load_peak_annotations(peak_gene_csv)
    else:
        peak_annot = None

    class_dropdown = widgets.Dropdown(
        options=classes,
        value=classes[0],
        description="Class:",
        layout=widgets.Layout(width=style_config["controls"]["class_width"]),
    )

    support_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.01,
        description="Min cell freq:",
        readout_format=".0%",
        style={"description_width": "auto"},
        layout=widgets.Layout(width=style_config["controls"]["support_width"]),
    )

    layout_dropdown = widgets.Dropdown(
        options=style_config["layout"]["available_layouts"],
        value=style_config["layout"]["default_layout"],
        description="Layout:",
        layout=widgets.Layout(width=style_config["controls"]["layout_width"]),
    )

    regulon_checkboxes = {}
    regulon_box = widgets.VBox(
        [],
        layout=widgets.Layout(
            width="100%",
            max_height=style_config["controls"].get("regulon_box_height", "260px"),
            overflow_y="auto",
            padding="4px 0 0 0",
        ),
    )

    select_all_btn = widgets.Button(description="All",  layout=widgets.Layout(width="60px"))
    clear_all_btn  = widgets.Button(description="None", layout=widgets.Layout(width="70px"))

    regulon_inner = widgets.VBox(
        [widgets.HBox([select_all_btn, clear_all_btn],
                      layout=widgets.Layout(gap="6px", align_items="center")),
         regulon_box],
        layout=widgets.Layout(width="100%"),
    )

    regulon_panel = widgets.Accordion(
        children=[regulon_inner],
        layout=widgets.Layout(width=style_config["controls"]["regulon_width"]),
    )
    regulon_panel.set_title(0, "TFs")
    regulon_panel.selected_index = None

    controls = widgets.HBox(
        [class_dropdown, regulon_panel, support_slider, layout_dropdown],
        layout=widgets.Layout(
            gap=style_config["controls"]["gap"],
            flex_flow="row wrap",
            align_items="flex-start",
        ),
    )

    graph_out  = widgets.Output(layout=widgets.Layout(width="100%"))
    info_panel = widgets.HTML(
        value=(
            f"<pre style='margin:0; white-space:pre-wrap; "
            f"font-family:{style_config['panels']['font_family']};'>"
            "Click a node or edge to inspect details.</pre>"
        ),
        layout=widgets.Layout(
            border=style_config["panels"]["border"],
            padding=style_config["panels"]["padding"],
            width="100%",
            min_height=style_config["panels"]["info_min_height"],
            overflow_y="auto",
        ),
    )

    sidebar = widgets.VBox(
        [info_panel, make_legend_panel(style_config, rna_only=rna_only)],
        layout=widgets.Layout(
            width=style_config["panels"]["sidebar_width"],
            min_width=style_config["panels"]["sidebar_width"],
            gap=style_config["panels"]["gap"],
            align_self="stretch",
        ),
    )

    main_row  = widgets.HBox(
        [graph_out, sidebar],
        layout=widgets.Layout(width="100%", align_items="stretch",
                              gap=style_config["panels"]["main_gap"]),
    )
    container = widgets.VBox(
        [controls, main_row],
        layout=widgets.Layout(
            width="100%",
            padding=style_config["panels"]["container_padding"],
            gap=style_config["panels"]["gap"],
        ),
    )

    def get_selected_regulons():
        return tuple(tf for tf, cb in regulon_checkboxes.items() if cb.value)

    def load_filtered_class(selected_class):
        raw = load_class_table(class_to_csv, selected_class)
        return filter_paths(raw, max_q_value=max_q_value, min_cell_freq=min_cell_freq,
                            min_n_cells=min_n_cells, top_n_paths=top_n_paths)

    def update_support_bounds(df):
        if len(df) == 0:
            support_slider.min = support_slider.max = support_slider.value = 0.0
            support_slider.step = 0.01
            return
        sup_min  = float(df["cell_freq"].fillna(0).min())
        sup_max  = float(df["cell_freq"].fillna(0).max())
        sup_step = max((sup_max - sup_min) / 100.0, 0.001)
        current  = float(support_slider.value)
        support_slider.min   = sup_min
        support_slider.max   = sup_max
        support_slider.step  = sup_step
        support_slider.value = min(max(current, sup_min), sup_max)

    def redraw(*args):
        selected_class    = class_dropdown.value
        selected_regulons = get_selected_regulons()
        min_support       = float(support_slider.value)
        layout_name       = layout_dropdown.value

        df = load_filtered_class(selected_class)

        if selected_regulons:
            df = df[df["tf_name"].astype(str).isin(selected_regulons)].copy()

        if max_tfs is not None and len(df) > 0:
            top_tfs = sorted(df["tf_name"].astype(str).value_counts().head(int(max_tfs)).index)
            df = df[df["tf_name"].astype(str).isin(top_tfs)].copy()

        edges = aggregate_edges(df, style_config)
        edges = edges[edges["support"] >= min_support].copy()

        connected = set(edges["source"]).union(set(edges["target"]))
        nodes     = build_nodes(df, peak_annot, style_config, edges)
        nodes     = nodes[nodes["id"].isin(connected)].copy()

        _set_info_panel(info_panel, format_selection_summary(),
                        font_family=style_config["panels"]["font_family"])

        with graph_out:
            graph_out.clear_output(wait=True)
            if len(nodes) == 0 or len(edges) == 0:
                display(widgets.HTML("<div>No graph to display for the current filters.</div>"))
                return
            graph = make_graph_widget(nodes, edges, style_config=style_config,
                                      layout_name=layout_name, info_panel=info_panel)
            display(graph)

    def update_regulon_options(df):
        nonlocal regulon_checkboxes
        old_selected = set(get_selected_regulons())
        regulons     = sorted(df["tf_name"].astype(str).unique())
        regulon_checkboxes = {}
        children = []
        for tf in regulons:
            cb = widgets.Checkbox(value=(tf in old_selected), description=tf,
                                  indent=False, layout=widgets.Layout(width="auto"))
            cb.observe(redraw, names="value")
            regulon_checkboxes[tf] = cb
            children.append(cb)
        regulon_box.children = tuple(children)

    def on_select_all(_):
        for cb in regulon_checkboxes.values():
            cb.unobserve(redraw, names="value"); cb.value = True; cb.observe(redraw, names="value")
        redraw()

    def on_clear_all(_):
        for cb in regulon_checkboxes.values():
            cb.unobserve(redraw, names="value"); cb.value = False; cb.observe(redraw, names="value")
        redraw()

    def on_class_change(change):
        if change["name"] == "value" and change["new"] is not None:
            df = load_filtered_class(change["new"])
            update_regulon_options(df)
            update_support_bounds(df)
            _set_info_panel(info_panel, format_selection_summary(),
                            font_family=style_config["panels"]["font_family"])
            redraw()

    select_all_btn.on_click(on_select_all)
    clear_all_btn.on_click(on_clear_all)
    class_dropdown.observe(on_class_change, names="value")
    support_slider.observe(redraw, names="value")
    layout_dropdown.observe(redraw, names="value")

    df0 = load_filtered_class(class_dropdown.value)
    update_regulon_options(df0)
    update_support_bounds(df0)

    display(container)
    redraw()