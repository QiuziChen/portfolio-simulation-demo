from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import folium
from matplotlib import colormaps as mpl_colormaps
from matplotlib.colors import to_hex
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sim.postal_sim import load_grid_and_population, resolve_path
from sim.simulation import SingleCompositionStats, load_precomputed_results

PRECOMPUTED_PATH = resolve_path("data/precomputed_results.pkl")


def _make_map(
    grid_gdf,
    mean_matrix,
    q05_matrix,
    q95_matrix,
) -> folium.Map:
    map_gdf = grid_gdf[["grid_id", "geometry"]].copy()
    map_gdf = map_gdf.to_crs("EPSG:4326")

    mean_layer = mean_matrix.rename(columns={"total_time": "mean_total_time"})
    q05_layer = q05_matrix.rename(columns={"total_time": "q05_total_time"})
    q95_layer = q95_matrix.rename(columns={"total_time": "q95_total_time"})

    map_gdf = map_gdf.merge(mean_layer[["grid_id", "mean_total_time"]], on="grid_id", how="left")
    map_gdf = map_gdf.merge(q05_layer[["grid_id", "q05_total_time"]], on="grid_id", how="left")
    map_gdf = map_gdf.merge(q95_layer[["grid_id", "q95_total_time"]], on="grid_id", how="left")
    map_gdf = map_gdf.fillna(0)

    center = map_gdf.geometry.centroid
    center_lat = float(center.y.mean())
    center_lon = float(center.x.mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)
    folium.TileLayer("CartoDB positron", name="Basemap", control=False).add_to(fmap)

    reds = mpl_colormaps["plasma"]

    def _add_layer(frame, value_column: str, layer_name: str, show: bool) -> None:
        layer_frame = frame[["grid_id", "geometry", value_column]].copy()
        layer_frame = layer_frame.fillna({value_column: 0.0})
        layer_frame = layer_frame[layer_frame[value_column] > 0].copy()
        values = layer_frame[value_column].to_numpy(dtype=float)
        min_value = float(values.min()) if len(values) else 0.0
        max_value = float(values.max()) if len(values) else 0.0
        scale = max(max_value - min_value, 1e-9)

        feature_group = folium.FeatureGroup(name=layer_name, overlay=False, control=True, show=show)

        def _style_function(feature):
            value = float(feature["properties"].get(value_column, 0.0))
            normalized = (value - min_value) / scale if scale > 0 else 0.0
            return {
                "fillColor": to_hex(reds(normalized)),
                "color": "#2f2f2f",
                "weight": 0.2,
                "fillOpacity": 0.7,
            }

        folium.GeoJson(
            layer_frame.to_json(),
            name=layer_name,
            style_function=_style_function,
            highlight_function=lambda feature: {"weight": 1.5, "color": "#ffffff"},
            tooltip=folium.GeoJsonTooltip(fields=["grid_id", value_column], aliases=["Grid", layer_name]),
        ).add_to(feature_group)

        feature_group.add_to(fmap)

    _add_layer(map_gdf, "mean_total_time", "Average Travel Time", show=True)
    _add_layer(map_gdf, "q05_total_time", "5th Quantile Utility Scenario", show=False)
    _add_layer(map_gdf, "q95_total_time", "95th Quantile Utility Scenario", show=False)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def _run_app() -> None:
    st.set_page_config(page_title="Sensing Portfolio Dashboard", layout="wide")
    st.title("Sensing Portfolio Dashboard")

    if not PRECOMPUTED_PATH.exists():
        st.error(
            "Precomputed results not found. "
            "Run `python precompute.py` locally and push `data/precomputed_results.pkl`."
        )
        return

    if "frontier" not in st.session_state or st.session_state.frontier is None:
        with st.spinner("Loading precomputed results…"):
            all_results, frontier, stats_map = load_precomputed_results(PRECOMPUTED_PATH)
        st.session_state.sim_results = all_results
        st.session_state.frontier = frontier
        st.session_state.stats_map = stats_map

    if st.session_state.frontier is not None and not st.session_state.frontier.empty:
        frontier = st.session_state.frontier.copy()
        frontier["composition_label"] = frontier.apply(
            lambda row: f"postal={int(row['n_postal'])}, ride={int(row['n_ridehailing'])}, total={int(row['total_vehicles'])}",
            axis=1,
        )

        st.subheader("Pareto Frontier (x: Expected Utility, y: 5th Quantile Utility)")
        st.caption("Click any point to select it.")

        if "pareto_selected_label" not in st.session_state or st.session_state.pareto_selected_label not in frontier["composition_label"].values:
            st.session_state.pareto_selected_label = frontier["composition_label"].iloc[0]

        selected_label = st.session_state.pareto_selected_label
        selected_row = frontier[frontier["composition_label"] == selected_label].iloc[0]
        selected_composition = (int(selected_row["n_postal"]), int(selected_row["n_ridehailing"]))

        fig = px.scatter(
            frontier,
            x="expected_utility",
            y="5th quantile utility",
            color="fleet size",
            hover_data=["n_postal", "n_ridehailing"],
            custom_data=["composition_label"],
            title="Pareto Frontier by Total Vehicles",
        )
        fig.update_layout(height=700)
        fig.update_traces(marker=dict(size=12, opacity=0.85), selector=dict(mode="markers"))
        fig.add_trace(go.Scatter(
            x=[selected_row["expected_utility"]],
            y=[selected_row["5th quantile utility"]],
            mode="markers",
            marker=dict(color="#d62728", size=22, symbol="star"),
            name="Selected composition",
            hovertemplate="Selected composition<extra></extra>",
        ))

        stats: SingleCompositionStats = st.session_state.stats_map[selected_composition]

        grid_gdf = load_grid_and_population()
        fmap = _make_map(
            grid_gdf=grid_gdf,
            mean_matrix=stats.mean_travel_time_matrix,
            q05_matrix=stats.q05_travel_time_matrix,
            q95_matrix=stats.q95_travel_time_matrix,
        )

        chart_col, utility_col = st.columns([2, 1])
        with chart_col:
            chart_event = st.plotly_chart(
                fig,
                on_select="rerun",
                selection_mode="points",
                key="pareto_chart",
                use_container_width=True,
            )
            if chart_event.selection.points:
                cd = chart_event.selection.points[0].get("customdata")
                if cd and cd[0] is not None:
                    clicked_label = str(cd[0])
                    if clicked_label != st.session_state.pareto_selected_label:
                        st.session_state.pareto_selected_label = clicked_label
                        st.rerun()

        with utility_col:
            st.subheader("Utility Summary for Selected Point")
            upper_error = max(float(stats.utility_q95 - stats.expected_utility), 0.0)
            lower_error = max(float(stats.expected_utility - stats.utility_q05), 0.0)
            scatter_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=["Expected utility"],
                        y=[stats.expected_utility],
                        mode="markers",
                        marker=dict(color="#1f77b4", size=14, symbol="circle"),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[upper_error],
                            arrayminus=[lower_error],
                            color="#444444",
                            thickness=1.5,
                            width=7,
                        ),
                    )
                ]
            )
            scatter_fig.update_layout(
                yaxis_title="Utility",
                xaxis_title="",
                showlegend=False,
                height=420,
                yaxis=dict(range=[
                    max(0.0, stats.utility_q05 - 0.01),
                    min(1.0, stats.utility_q95 + 0.01),
                ]),
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        st.subheader("Travel Time Distribution Map")
        st.components.v1.html(fmap._repr_html_(), height=650)

        st.subheader("Selected Composition Details")
        st.write(
            {
                "composition": {
                    "n_postal": stats.n_postal,
                    "n_ridehailing": stats.n_ridehailing,
                    "fleet_size": stats.total_vehicles,
                },
                "avg_utility": stats.expected_utility,
                "5th_quantile_utility": stats.utility_q05,
                "95th_quantile_utility": stats.utility_q95,
            }
        )
    elif st.session_state.get("frontier") is not None:
        st.warning("No feasible frontier points were found.")
